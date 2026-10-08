"""
FastAPI service for the RAG pipeline.
========================================

Three endpoints, all reusing Phase 1-4 exactly as built -- this file adds
no new RAG logic, only an HTTP surface over generation.rag_answer(),
Qdrant.list_documents(), and Qdrant.ingest_strategy_chunks().

    POST /v1/ask        question in, cited answer + confidence + sources out
    GET  /v1/documents   what's actually indexed right now, per strategy
    POST /v1/ingest      submit new chunks for indexing

OpenAPI docs are automatic: FastAPI generates them from the Pydantic models
and docstrings below. Once running: /docs (Swagger UI) and /redoc.

Run locally
    uvicorn api:app --reload --port 8000

Run in docker-compose
    see docker-compose.yml / Dockerfile.api
"""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import generation
import Qdrant
import retrieval

app = FastAPI(
    title="RAG Pipeline API",
    description=(
        "Hybrid retrieval (dense + BM25 + RRF + cross-encoder rerank) with "
        "grounded, cited generation and an automated confidence score. "
        "Backed by ChromaDB, BM25, and a local Ollama model."
    ),
    version="1.0.0",
)

# Lets the Streamlit/React dashboard call this API from a different origin
# (different container, different port) without the browser blocking it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================== POST /v1/ask
class AskRequest(BaseModel):
    question: str = Field(..., description="The user's natural-language question.", min_length=1)
    strategy: str = Field(..., description="Which chunking strategy's index to search, e.g. 'recursive_512'.")
    use_sparse: bool = Field(True, description="False = dense-only retrieval (skips BM25 + RRF). Used for hybrid-vs-dense-only comparison.")
    retrieval_threshold: float = Field(
        generation.RETRIEVAL_CONFIDENCE_THRESHOLD,
        description="Below this retrieval confidence, the system refuses to answer rather than risk hallucinating.",
        ge=0.0, le=1.0,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What config key controls the retry timeout?",
                "strategy": "recursive_512",
                "use_sparse": True,
                "retrieval_threshold": 0.35,
            }
        }


class CitationCheck(BaseModel):
    n: int = Field(..., description="Citation number as it appears in the answer, e.g. [2].")
    chunk_id: Optional[str] = None
    supported: bool = Field(..., description="Whether an LLM judge confirmed this chunk actually backs the claim.")
    reason: str = ""


class ClaimVerification(BaseModel):
    sentence: str
    citations: list[int] = Field(default_factory=list, description="Citation numbers this sentence used, e.g. [1, 2].")
    is_disclaimer: bool = Field(..., description="True if this sentence is the model honestly saying it doesn't know, not a claim.")
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    covered: Optional[bool] = Field(None, description="True if at least one citation checked out. None for disclaimer sentences.")


class ConfidenceBreakdown(BaseModel):
    retrieval_confidence: float = Field(..., description="How relevant the top retrieved chunks were (0-1).")
    citation_coverage: Optional[float] = Field(None, description="Fraction of claims backed by a verified citation. None if the answer was a pure refusal.")
    completeness: Optional[float] = Field(None, description="Did the answer address every part of the question (0-1)?")
    composite: float = Field(..., description="Weighted blend of the above -- the single number to sort/filter answers by.")


class SourceChunk(BaseModel):
    n: int = Field(..., description="Citation number this chunk corresponds to in the answer, e.g. [1].")
    id: str
    source_document: str
    section_heading: str


class AskResponse(BaseModel):
    query: str
    status: Literal["answered", "insufficient_context"]
    answer: Optional[str] = Field(None, description="The generated, cited answer. None if the system refused to answer.")
    message: Optional[str] = Field(None, description="Explanation shown only when status is insufficient_context.")
    sources: list[SourceChunk] = Field(default_factory=list, description="The chunks the answer is grounded in, numbered to match its [n] citations.")
    citation_verification: list[ClaimVerification] = Field(default_factory=list)
    suggested_documents_to_check: list[str] = Field(default_factory=list, description="Only populated when status is insufficient_context.")
    confidence: ConfidenceBreakdown


@app.post("/v1/ask", response_model=AskResponse, tags=["ask"])
def ask(req: AskRequest) -> AskResponse:
    """
    Answers a question against one chunking strategy's index. Runs the full
    Phase 2 (hybrid retrieval) + Phase 3 (grounded generation, citation
    verification, confidence scoring) pipeline.

    If retrieval confidence falls below `retrieval_threshold`, the system
    returns `status: "insufficient_context"` instead of generating anything
    -- see `suggested_documents_to_check` for where to look manually.
    """
    if req.strategy not in Qdrant.list_strategies():
        raise HTTPException(status_code=404, detail=f"Unknown strategy {req.strategy!r}. See GET /v1/documents for available strategies.")

    result = generation.rag_answer(
        req.question, req.strategy,
        retrieval_threshold=req.retrieval_threshold,
        use_sparse=req.use_sparse,
    )

    if result["status"] == "insufficient_context":
        sources = [
            SourceChunk(n=i, id=m["id"], source_document=m["source_document"], section_heading=m["section_heading"])
            for i, m in enumerate(result["closest_matches_found"], start=1)
        ]
        return AskResponse(
            query=result["query"], status=result["status"], message=result["message"],
            sources=sources, suggested_documents_to_check=result["suggested_documents_to_check"],
            confidence=ConfidenceBreakdown(**result["confidence"]),
        )

    sources = [
        SourceChunk(n=c["n"], id=c["id"], source_document=c["source_document"], section_heading=c["section_heading"])
        for c in result["chunks_used"]
    ]
    return AskResponse(
        query=result["query"], status=result["status"], answer=result["answer"],
        sources=sources, citation_verification=result["citation_verification"],
        confidence=ConfidenceBreakdown(**result["confidence"]),
    )


# ===================================================== GET /v1/documents
class DocumentInfo(BaseModel):
    source_document: str
    chunking_strategy: str
    chunk_count: int


@app.get("/v1/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents() -> list[DocumentInfo]:
    """
    Lists every (document, chunking strategy) pair currently indexed, read
    straight from Chroma -- this can never drift from what's actually
    searchable, unlike a separately maintained document registry would.
    """
    return [DocumentInfo(**row) for row in Qdrant.list_documents()]


@app.get("/v1/strategies", response_model=list[str], tags=["documents"])
def list_strategies() -> list[str]:
    """Every chunking strategy that has its own index right now (e.g. ['recursive_512', 'semantic'])."""
    return Qdrant.list_strategies()


# ========================================================= POST /v1/ingest
class IngestChunk(BaseModel):
    text: str = Field(..., min_length=1)
    source_document: str
    chunk_index: int = Field(..., ge=0)
    section_heading: str = ""
    chunking_strategy: str = Field(..., description="Which strategy's index this chunk belongs to. New strategies are created automatically.")


class IngestRequest(BaseModel):
    chunks: list[IngestChunk] = Field(..., min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "chunks": [
                    {
                        "text": "The retry_timeout_ms config key controls how long to wait before retrying.",
                        "source_document": "api_reference.md",
                        "chunk_index": 0,
                        "section_heading": "Retries",
                        "chunking_strategy": "recursive_512",
                    }
                ]
            }
        }


class StrategyIngestResult(BaseModel):
    strategy: str
    chunks_submitted: int
    chunks_indexed: int
    duplicates_skipped: int = Field(..., description="Near-duplicates (cosine > 0.95) that were skipped rather than indexed.")


class IngestResponse(BaseModel):
    results: list[StrategyIngestResult]


@app.post("/v1/ingest", response_model=IngestResponse, tags=["ingest"])
def ingest(req: IngestRequest) -> IngestResponse:
    """
    Indexes new chunks: embeds them, checks for near-duplicates against
    what's already stored, writes to Chroma, and rebuilds BM25 from Chroma
    so the two indexes never drift apart (see Qdrant.py). Chunks are
    grouped by `chunking_strategy` and each group is ingested into its own
    isolated index -- a new strategy name just creates a new index.
    """
    domain_chunks = [Qdrant.Chunk(**c.model_dump()) for c in req.chunks]
    by_strategy = Qdrant.group_by_strategy(domain_chunks)

    results = [
        StrategyIngestResult(
            strategy=r["strategy"], chunks_submitted=r["chunks_submitted"],
            chunks_indexed=r["chunks_indexed"], duplicates_skipped=r["duplicates_skipped"],
        )
        for strategy, group in by_strategy.items()
        for r in [Qdrant.ingest_strategy_chunks(group, strategy)]
    ]
    return IngestResponse(results=results)


# ------------------------------------------------------------------ health
@app.get("/health", tags=["health"])
def health() -> dict:
    """Liveness check for docker-compose / load balancers. Does not verify Ollama or Chroma connectivity."""
    return {"status": "ok"}
