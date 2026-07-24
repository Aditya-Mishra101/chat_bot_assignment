"""Preload models and clients at startup to avoid cold-start latency on first request."""

import logging

from app.core.config import settings

logger = logging.getLogger("rag_warmup")


def warmup_models() -> None:
    from app.core.embedding import get_embedding_model, get_sparse_model, get_embeddings, get_sparse_embeddings
    from app.core.llm import get_llm, get_optimizer_llm
    from app.core.reranker import get_reranker_model
    from app.core.vector_store import collection_count, get_qdrant_client

    logger.info("Warming up RAG pipeline components...")

    get_embedding_model()
    get_sparse_model()
    get_qdrant_client()
    get_reranker_model()
    get_optimizer_llm()
    get_llm(settings.DEFAULT_LLM_BACKEND)

    get_embeddings(["warmup"])
    get_sparse_embeddings(["warmup"])
    collection_count(settings.COLLECTION_NAME)

    logger.info("RAG pipeline warmup complete.")
