from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings 
from langchain.schema import Document
import os
from datetime import datetime, timezone
import uuid
import chromadb

import argparse
import json
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import requests
from rank_bm25 import BM25Okapi
 
# ----------------------------------------------------------------- config
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"      # run `ollama pull nomic-embed-text` first
EMBED_BATCH = 100                     # chunks per request
CHROMA_DIR = "./chroma_db"
COLLECTION = "chunks"
BM25_PATH = Path("./bm25_index.pkl")
DEDUP_THRESHOLD = 0.95                # cosine similarity above this = duplicate
 
# ------------------------------------------------------------------ model
@dataclass
class Chunk:
    text: str
    source_document: str
    chunk_index: int
    section_heading: str
    chunking_strategy: str
 
    @property
    def id(self) -> str:
        # Deterministic ID: re-running the pipeline upserts, never duplicates.
        return f"{self.source_document}::{self.chunking_strategy}::{self.chunk_index}"
 
    @property
    def metadata(self) -> dict:
        # Chroma metadata values must be str/int/float/bool (no None).
        return {
            "source_document": self.source_document,
            "chunk_index": self.chunk_index,
            "section_heading": self.section_heading or "",
            "chunking_strategy": self.chunking_strategy,
            "char_count": len(self.text),
        }
 
 
def load_chunks(path: str) -> list[Chunk]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks
 
 
# -------------------------------------------------------------- embedding
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed via a local Ollama server's /api/embed endpoint.
    Ollama must be running (`ollama serve`) with EMBED_MODEL pulled.
    """
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        for attempt in range(5):
            try:
                resp = requests.post(
                    f"{OLLAMA_HOST}/api/embed",
                    json={"model": EMBED_MODEL, "input": batch},
                    timeout=120,
                )
                resp.raise_for_status()
                vectors.extend(resp.json()["embeddings"])  # order preserved
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 2**attempt
                print(f"  embed error ({e}); retrying in {wait}s")
                time.sleep(wait)
        print(f"  embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
    return vectors
 
 
def embed_query(text: str) -> list[float]:
    """Embed a single query string the same way (used by search())."""
    return embed_texts([text])[0]
 
 
# ---------------------------------------------------------------- chroma
def get_collection():
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    return db.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # cosine suits normalized embedding vectors
    )
 
 
def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
 
 
def closest_existing_match(vector: list[float], col) -> tuple[str, float] | None:
    """Nearest neighbour already stored in Chroma, or None if collection is empty."""
    if col.count() == 0:
        return None
    res = col.query(query_embeddings=[vector], n_results=1, include=["distances"])
    if not res["ids"][0]:
        return None
    # hnsw:space="cosine" -> Chroma's distance is (1 - cosine_similarity)
    return res["ids"][0][0], 1 - res["distances"][0][0]
 
 
def dedupe_chunks(
    chunks: list[Chunk], vectors: list[list[float]], col
) -> tuple[list[Chunk], list[list[float]], list[dict]]:
    """
    Flag and drop near-duplicates before they ever reach Chroma/BM25.
    Checks each new chunk against:
      1) chunks already stored in Chroma (previous runs), and
      2) chunks already accepted earlier in this same batch,
    so two duplicates arriving in the same file are also caught.
    """
    kept_chunks, kept_vectors, dropped = [], [], []
    batch_ids: list[str] = []
    batch_vectors: list[list[float]] = []
 
    for chunk, vector in zip(chunks, vectors):
        best_id, best_sim = None, -1.0
 
        existing = closest_existing_match(vector, col)
        if existing:
            best_id, best_sim = existing
 
        for bid, bvec in zip(batch_ids, batch_vectors):
            sim = cosine_sim(vector, bvec)
            if sim > best_sim:
                best_id, best_sim = bid, sim
 
        if best_sim > DEDUP_THRESHOLD:
            dropped.append({"id": chunk.id, "duplicate_of": best_id, "similarity": round(best_sim, 4)})
        else:
            kept_chunks.append(chunk)
            kept_vectors.append(vector)
            batch_ids.append(chunk.id)
            batch_vectors.append(vector)
 
    return kept_chunks, kept_vectors, dropped
 
 
def upsert_to_chroma(chunks: list[Chunk]) -> list[dict]:
    col = get_collection()
 
    # If a document is re-ingested with fewer chunks, old ones would linger.
    # Remove every existing chunk for these (document, strategy) pairs first,
    # so this exact document is never compared against its own old version.
    for src, strat in {(c.source_document, c.chunking_strategy) for c in chunks}:
        col.delete(where={"$and": [
            {"source_document": src},
            {"chunking_strategy": strat},
        ]})
 
    vectors = embed_texts([c.text for c in chunks])
 
    chunks, vectors, duplicates = dedupe_chunks(chunks, vectors, col)
    if duplicates:
        print(f"Skipped {len(duplicates)} near-duplicate chunk(s) (cosine > {DEDUP_THRESHOLD}):")
        for d in duplicates:
            print(f"  {d['id']}  ~=  {d['duplicate_of']}  (sim={d['similarity']})")
 
    if chunks:
        col.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[c.metadata for c in chunks],
        )
    print(f"Chroma now holds {col.count()} chunks")
    return duplicates
 
 
def read_all_from_chroma() -> tuple[list[str], list[str]]:
    """Return (ids, texts) for the whole collection, paginated."""
    col = get_collection()
    ids, texts, offset, page = [], [], 0, 1000
    while True:
        res = col.get(include=["documents"], limit=page, offset=offset)
        if not res["ids"]:
            break
        ids.extend(res["ids"])
        texts.extend(res["documents"])
        offset += page
    order = sorted(range(len(ids)), key=lambda i: ids[i])   # stable ordering
    return [ids[i] for i in order], [texts[i] for i in order]
 
 
# ------------------------------------------------------------------ bm25
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
 
 
def rebuild_bm25() -> None:
    """Rebuild BM25 from Chroma's contents -> the two indexes cannot diverge."""
    ids, texts = read_all_from_chroma()
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    # Store ids and texts as-is (no hashing) so verify_sync() can compare
    # them directly against Chroma's current contents.
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"ids": ids, "texts": texts, "bm25": bm25}, f)
    print(f"BM25 rebuilt over {len(ids)} chunks")
 
 
def load_bm25() -> dict:
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)
 
 
# ------------------------------------------------------------ sync check
def verify_sync() -> bool:
    """
    Compares Chroma's current (ids, texts) directly against what BM25 was
    built from -- plain list/string equality, no hashing.
    """
    ids, texts = read_all_from_chroma()
    bm = load_bm25()
    ok = True
 
    if bm["ids"] != ids:
        missing = set(ids) - set(bm["ids"])
        extra = set(bm["ids"]) - set(ids)
        print(f"MISMATCH ids: {len(missing)} missing from BM25, {len(extra)} extra")
        ok = False
 
    if bm["texts"] != texts:
        # ids line up but content differs somewhere -> find exactly where
        bm_lookup = dict(zip(bm["ids"], bm["texts"]))
        changed = [i for i, t in zip(ids, texts) if bm_lookup.get(i) != t]
        print(f"MISMATCH: {len(changed)} chunk(s) changed in Chroma since BM25 was built: {changed[:5]}")
        ok = False
 
    print("Indexes are in sync" if ok else "Run `build` again to resync")
    return ok
 
 
# ---------------------------------------------------------------- driver
def build(path: str) -> None:
    chunks = load_chunks(path)
    print(f"Loaded {len(chunks)} chunks")
    duplicates = upsert_to_chroma(chunks)   # 1) dense index (source of truth), deduped
    rebuild_bm25()                          # 2) sparse index derived from step 1
    assert verify_sync(), "Indexes out of sync after build"
    if duplicates:
        print(f"\n{len(duplicates)} chunk(s) were skipped as near-duplicates; "
              f"{len(chunks) - len(duplicates)} unique chunks are indexed.")
 
 
def search(query: str, k: int = 5) -> None:
    """Quick side-by-side look at both retrievers."""
    col = get_collection()
    q_vec = embed_query(query)
    dense = col.query(query_embeddings=[q_vec], n_results=k,
                      include=["documents", "metadatas", "distances"])
 
    bm = load_bm25()
    scores = bm["bm25"].get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
 
    print("\n== Dense (Chroma) ==")
    for cid, doc, dist in zip(dense["ids"][0], dense["documents"][0], dense["distances"][0]):
        print(f"{1 - dist:.3f}  {cid}\n       {doc[:90]!r}")
    print("\n== Sparse (BM25) ==")
    for i in top:
        print(f"{scores[i]:.3f}  {bm['ids'][i]}")
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("chunks_file")
    sub.add_parser("verify")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    a = p.parse_args()
 
    if a.cmd == "build":
        build(a.chunks_file)
    elif a.cmd == "verify":
        verify_sync()
    else:
        search(a.query, a.k)