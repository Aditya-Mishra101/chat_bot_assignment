import time
import logging
from app.core.config import settings
from app.core.embedding import get_query_embedding, get_query_sparse_embedding
from app.core.vector_store import search_chunks
from app.core.llm import NO_CONTEXT_TOKEN, generate_answer, generate_streaming

logger = logging.getLogger("rag_retrieval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def answer_query(query: str, llm_backend_override: str = None, top_k: int = settings.CHUNK, stream: bool = False) -> dict:
    start_time = time.perf_counter()
    embed_start = time.perf_counter()
    query_embedding = get_query_embedding(query)
    query_sparse_embedding = get_query_sparse_embedding(query)
    embed_time = time.perf_counter() - embed_start

    retrieve_start = time.perf_counter()
    sources = search_chunks(
        collection_name=settings.COLLECTION_NAME,
        query_embedding=query_embedding,
        top_k=top_k,
        query_sparse_embedding=query_sparse_embedding,
    )
    retrieve_time = time.perf_counter() - retrieve_start

    context_texts = []
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata", {})
        source_name = meta.get("source", "Unknown")
        page = meta.get("page", "")
        context_texts.append(f"--- Chunk {i} ---\nSource: {source_name} {f'Page: {page}' if page else ''}\n{s['text']}\n")
    context = "\n".join(context_texts)

    llm_start = time.perf_counter()
    backend_to_use = llm_backend_override or settings.DEFAULT_LLM_BACKEND

    if stream:
        return {
            "answer": generate_streaming(query, context, backend_to_use),
            "sources": sources,
            "llm_backend_used": backend_to_use,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
    else:
        answer = generate_answer(query, context, backend_to_use)

    if NO_CONTEXT_TOKEN in answer.strip().upper():
        answer = "I could not find any relevant information in the provided documents to answer your question."
        sources = []

    llm_time = time.perf_counter() - llm_start
    total_time = time.perf_counter() - start_time
    total_time_ms = round(total_time * 1000, 2)

    logger.info(f"Query: '{query}' | Backend: {backend_to_use} | Chunks retrieved: {len(sources)}")
    logger.info(f"Latency -> Embed: {embed_time*1000:.2f}ms, Retrieve: {retrieve_time*1000:.2f}ms, LLM: {llm_time*1000:.2f}ms, Total: {total_time_ms}ms")
    for s in sources:
        logger.info(f"Retrieved Score: {s['similarity_score']:.4f} | Source: {s.get('metadata', {}).get('source')}")

    return {
        "answer": answer,
        "sources": sources,
        "llm_backend_used": backend_to_use,
        "latency_ms": total_time_ms
    }