"""
Personalized Search Result Re-Ranking Backend
Uses Sentence-BERT (all-MiniLM-L6-v2) to semantically re-rank Google search results.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import logging

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Search Re-Ranking API",
    description="Semantically re-ranks Google search results using Sentence-BERT",
    version="1.0.0",
)

# Allow all origins so the Chrome extension can reach this local server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Loading (once at startup) ──────────────────────────────────────────
# Set MODEL_PATH env var to load your fine-tuned model instead of the base one.
# Example: MODEL_PATH=./model_finetuned uvicorn main:app --port 8000
import os
MODEL_PATH = os.environ.get("MODEL_PATH", "all-MiniLM-L6-v2")
model: SentenceTransformer | None = None


@app.on_event("startup")
def load_model():
    """
    Load the SBERT model once when the server starts.
    Uses MODEL_PATH env var — defaults to base all-MiniLM-L6-v2.
    After fine-tuning, restart with MODEL_PATH=./model_finetuned.
    """
    global model
    logger.info(f"Loading SentenceTransformer model from: {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
    logger.info("Model loaded successfully.")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class SearchResult(BaseModel):
    title: str
    snippet: str
    url: str


class RerankRequest(BaseModel):
    query: str
    results: List[SearchResult]


class RankedResult(BaseModel):
    title: str
    snippet: str
    url: str
    score: float


class RerankResponse(BaseModel):
    ranked: List[RankedResult]


# ── Core Re-Ranking Logic ─────────────────────────────────────────────────────
def build_document_text(result: SearchResult) -> str:
    """Combine title and snippet into a single string for embedding."""
    return f"{result.title}. {result.snippet}"


def rerank_results(query: str, results: List[SearchResult]) -> List[RankedResult]:
    """
    Compute cosine similarity between the query embedding and each result embedding,
    then sort results from most to least semantically similar.
    """
    if not results:
        return []

    # Build text representations
    documents = [build_document_text(r) for r in results]

    # Generate embeddings for query and all documents in one batched call
    all_texts = [query] + documents
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)

    query_embedding = embeddings[0:1]          # shape (1, dim)
    doc_embeddings  = embeddings[1:]           # shape (N, dim)

    # Cosine similarity: query vs. each document → shape (1, N)
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Attach scores and sort descending
    scored = [
        RankedResult(
            title=results[i].title,
            snippet=results[i].snippet,
            url=results[i].url,
            score=float(similarities[i]),
        )
        for i in range(len(results))
    ]
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    """
    Accepts a search query + list of results, returns them re-ranked by
    semantic similarity to the query using SBERT cosine similarity.

    Falls back to the original order if anything goes wrong.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not yet loaded.")

    try:
        ranked = rerank_results(request.query, request.results)
        logger.info(
            f"Re-ranked {len(ranked)} results for query: '{request.query}'"
        )
        return RerankResponse(ranked=ranked)

    except Exception as exc:
        # Return original order with zero scores so the extension degrades gracefully
        logger.error(f"Re-ranking failed: {exc}")
        fallback = [
            RankedResult(
                title=r.title,
                snippet=r.snippet,
                url=r.url,
                score=0.0,
            )
            for r in request.results
        ]
        return RerankResponse(ranked=fallback)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
