import logging
from app.core.config import settings
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag_reranker")

_reranker_model = None

def get_reranker_model() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(settings.RE_RANKER_MODEL)
    return _reranker_model


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_n: int = settings.RERANK_TOP_N,
) -> list[dict]:
    """
    Re-ranks retrieved chunks against the query using a cross-encoder
    and returns the top_n most relevant.
    """
    if not chunks:
        return []

    pairs = [(query, c["text"]) for c in chunks]
    model = get_reranker_model()
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]