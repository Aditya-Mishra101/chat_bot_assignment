import logging
from app.core.config import settings
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag_reranker")

_reranker_model = None

def get_reranker_model() -> CrossEncoder:
    """Loaded once and cached — loading a CrossEncoder from disk/HF on every
    request would badly hurt latency."""
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    return _reranker_model


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = settings.RERANK_TOP_N,
    score_threshold: float = settings.RERANK_SCORE_THRESHOLD,
) -> list[dict]:
    """
    Re-ranks retrieved chunks against the query using a cross-encoder,
    filters out chunks below the score threshold, and returns only the
    top_n most relevant.

    `chunks` is expected to be the list of dicts returned by search_chunks:
    [{"text": ..., "metadata": ..., "similarity_score": ...}, ...]
    """
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]
    model = get_reranker_model()
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # ── Filter by score threshold ──
    filtered = [c for c in chunks if c["rerank_score"] >= score_threshold]

    if settings.DEBUG:
        dropped = len(chunks) - len(filtered)
        if dropped:
            logger.info(
                f"[DEBUG] Reranker dropped {dropped}/{len(chunks)} chunks "
                f"below threshold {score_threshold}"
            )

    # ── Sort and trim ──
    reranked = sorted(filtered, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]