
from __future__ import annotations
 
import argparse
import re
 
import requests
 
import chunking  # reuses get_collection(), load_bm25(), tokenize(), embed_query()
 
# ----------------------------------------------------------------- config
RRF_K = 60                   # standard RRF damping constant
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
FUSION_TOP_N = 20            # candidates passed into the reranker
FINAL_TOP_N = 5              # chunks returned after reranking
 
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # small, CPU-friendly
LLM_JUDGE_MODEL = "llama3.1"  # only used if rerank(method="llm")
 
 
# --------------------------------------------------------------- 1) dense
def dense_search(query: str, k: int = 10) -> list[dict]:
    """
    Embed the query and ask Chroma for the k nearest chunks by cosine
    similarity. Returns [{id, score, rank}], rank 1 = best match.
    """
    q_vec = chunking.embed_query(query)
    col = chunking.get_collection()
    res = col.query(query_embeddings=[q_vec], n_results=k, include=["distances"])
 
    results = []
    for rank, (cid, dist) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1):
        results.append({"id": cid, "score": 1 - dist, "rank": rank})  # cosine space
    return results
 
 
# -------------------------------------------------------------- 2) sparse
def sparse_search(query: str, k: int = 10) -> list[dict]:
    """
    Run the same query through BM25 over the chunk corpus. Returns the
    same shape as dense_search(): [{id, score, rank}].
    Catches exact matches (function names, config keys, error codes) that
    embeddings can blur together.
    """
    bm = chunking.load_bm25()
    scores = bm["bm25"].get_scores(chunking.tokenize(query))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
 
    results = []
    for rank, i in enumerate(top_idx, start=1):
        results.append({"id": bm["ids"][i], "score": float(scores[i]), "rank": rank})
    return results
 
 
# --------------------------------------------------------------- 3) fuse
def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    rrf_k: int = RRF_K,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
) -> list[dict]:
    """
    RRF score for a chunk = weight * 1 / (rrf_k + rank_in_that_list).
    A chunk found by both retrievers gets both contributions added together,
    so it naturally outranks a chunk only one method found. Chunks that
    appear in only one list still enter the merged ranking.
 
    rrf_k is a damping constant (60 is the standard value from the original
    RRF paper): it flattens the gap between rank 1 and rank 2, so one
    retriever's noisy top pick can't completely dominate the merge.
    dense_weight / sparse_weight make this tunable per use case, e.g. push
    sparse_weight higher for a corpus full of exact config keys and error
    codes, or dense_weight higher for conceptual, paraphrase-heavy queries.
    """
 
    fused_scores: dict[str, float] = {}
 
    for r in dense_results:
        fused_scores[r["id"]] = fused_scores.get(r["id"], 0.0) + dense_weight / (rrf_k + r["rank"])
 
    for r in sparse_results:
        fused_scores[r["id"]] = fused_scores.get(r["id"], 0.0) + sparse_weight / (rrf_k + r["rank"])
 
    ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"id": cid, "rrf_score": score} for cid, score in ranked]
 
def hydrate(ids: list[str]) -> dict[str, dict]:
    """
    RRF only knows ids and scores. This pulls the actual text + metadata
    for a specific set of ids straight from Chroma (the source of truth),
    so we never drift from what's actually indexed.
    """
    if not ids:
        return {}
    col = chunking.get_collection()
    res = col.get(ids=ids, include=["documents", "metadatas"])
    return {
        cid: {"text": doc, "metadata": meta}
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])
    }
 
 
# ------------------------------------------------------------- 4) rerank
_cross_encoder = None  # lazy-loaded so the model only loads once per process
 
 
def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading cross-encoder ({CROSS_ENCODER_MODEL})...")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder
 
 
def rerank_cross_encoder(query: str, candidates: list[dict], top_n: int = FINAL_TOP_N) -> list[dict]:
    """
    A cross-encoder reads (query, chunk) TOGETHER through one model and
    outputs a single relevance score -- unlike the dense/sparse passes,
    which score the query and chunk separately and compare vectors
    afterwards. That joint attention is slower (can't precompute chunk
    vectors ahead of time) but far more precise, which is why it only runs
    on the fused top 20 rather than the whole corpus.
    """
    model = _get_cross_encoder()
    pairs = [[query, c["text"]] for c in candidates]
    scores = model.predict(pairs)
 
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
 
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]
 
 
def _extract_score(text: str) -> float:
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else 0.0
 
 
def rerank_llm(query: str, candidates: list[dict], top_n: int = FINAL_TOP_N) -> list[dict]:
    """
    LLM-as-judge alternative to the cross-encoder: ask a local Ollama model
    to rate relevance 0-10 for each candidate. More flexible (you can put
    whatever judging criteria you want in the prompt) but slower and less
    consistent than a purpose-trained cross-encoder, since the model has to
    generate text rather than just output a score.
    """
    for c in candidates:
        prompt = (
            "Rate how relevant this passage is to the question, on a scale "
            "of 0 to 10. Respond with ONLY the number, nothing else.\n\n"
            f"Question: {query}\n\nPassage: {c['text']}\n\nRelevance score:"
        )
        resp = requests.post(
            f"{chunking.OLLAMA_HOST}/api/generate",
            json={"model": LLM_JUDGE_MODEL, "prompt": prompt, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        c["rerank_score"] = _extract_score(resp.json()["response"])
 
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_n]
 
 
def rerank(query: str, candidates: list[dict], top_n: int = FINAL_TOP_N, method: str = "cross_encoder") -> list[dict]:
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_n)
    elif method == "llm":
        return rerank_llm(query, candidates, top_n)
    raise ValueError(f"unknown rerank method: {method}")
 
 
# --------------------------------------------------------- the full engine
def hybrid_retrieve(
    query: str,
    dense_k: int = 10,
    sparse_k: int = 10,
    fusion_top_n: int = FUSION_TOP_N,
    final_top_n: int = FINAL_TOP_N,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    rerank_method: str = "cross_encoder",
) -> list[dict]:
    """
    Runs the whole Phase 2 pipeline end to end and returns the final
    chunks, each with id, text, metadata, rrf_score and rerank_score.
    """
    dense_results = dense_search(query, dense_k)
    sparse_results = sparse_search(query, sparse_k)
 
    fused = reciprocal_rank_fusion(dense_results, sparse_results, dense_weight=dense_weight, sparse_weight=sparse_weight)
    top_fused = fused[:fusion_top_n]
 
    content = hydrate([f["id"] for f in top_fused])
    candidates = [
        {"id": f["id"], "rrf_score": f["rrf_score"], **content[f["id"]]}
        for f in top_fused
        if f["id"] in content  # a stale id would be silently skipped rather than crash
    ]
 
    return rerank(query, candidates, top_n=final_top_n, method=rerank_method)
 
 
# --------------------------------------------------------------------- cli
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--rerank", choices=["cross_encoder", "llm"], default="cross_encoder")
    args = p.parse_args()
 
    results = hybrid_retrieve(args.query, rerank_method=args.rerank)
    print(f"\nTop {len(results)} chunks for: {args.query!r}\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. [{r['id']}]  rrf={r['rrf_score']:.4f}  rerank={r['rerank_score']:.3f}")
        print(f"   {r['text'][:120]!r}\n")