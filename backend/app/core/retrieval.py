import asyncio
import time
import logging
from app.core.config import settings
from app.core.embedding import get_query_embedding, get_query_sparse_embedding
from app.core.vector_store import search_chunks
from app.core.reranker import rerank_chunks
from app.core.llm import (
    NO_CONTEXT_TOKEN,
    generate_answer,
    generate_streaming,
)
from app.core.query_optimizer import optimize_query

logger = logging.getLogger("rag_retrieval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Deduplicate chunks by text content, keeping the highest similarity score."""
    seen = {}
    for c in chunks:
        text = c["text"]
        if text not in seen or c["similarity_score"] > seen[text]["similarity_score"]:
            seen[text] = c
    return list(seen.values())


async def answer_query(
    query: str,
    llm_backend_override: str = None,
    retrieve_k: int = None,
    rerank_top_n: int = None,
    stream: bool = False,
) -> dict:
    start_time = time.perf_counter()
    debug_info = {} if settings.DEBUG else None

    # Resolve config defaults
    retrieve_k = retrieve_k or settings.RETRIEVE_K
    rerank_top_n = rerank_top_n or settings.RERANK_TOP_N
    backend_to_use = llm_backend_override or settings.DEFAULT_LLM_BACKEND

    # ── Query optimization ──
    opt_start = time.perf_counter()
    opt_result = await optimize_query(query)
    opt_time = time.perf_counter() - opt_start

    queries_to_run = opt_result["queries"]
    hyde_text = opt_result["hyde_text"]

    if settings.DEBUG:
        debug_info["query_optimization"] = opt_result["debug_info"]
        debug_info["query_optimization"]["latency_ms"] = round(opt_time * 1000, 2)

    # ── Embedding ──
    embed_start = time.perf_counter()

    # Embed all queries (original + expanded)
    all_embeddings = []
    all_sparse_embeddings = []
    for q in queries_to_run:
        all_embeddings.append(get_query_embedding(q))
        all_sparse_embeddings.append(get_query_sparse_embedding(q))

    # If HyDE is active, also embed the hypothetical passage
    if hyde_text:
        all_embeddings.append(get_query_embedding(hyde_text))
        all_sparse_embeddings.append(get_query_sparse_embedding(hyde_text))

    embed_time = time.perf_counter() - embed_start

    # ── Retrieval (run for each query, then deduplicate) ──
    retrieve_start = time.perf_counter()

    all_candidates = []
    for emb, sparse_emb in zip(all_embeddings, all_sparse_embeddings):
        candidates = search_chunks(
            collection_name=settings.COLLECTION_NAME,
            query_embedding=emb,
            top_k=retrieve_k,
            query_sparse_embedding=sparse_emb,
        )
        all_candidates.extend(candidates)

    # Deduplicate across query variants
    candidates = _deduplicate_chunks(all_candidates)
    retrieve_time = time.perf_counter() - retrieve_start

    if settings.DEBUG:
        debug_info["retrieval"] = {
            "queries_run": len(all_embeddings),
            "total_candidates_before_dedup": len(all_candidates),
            "candidates_after_dedup": len(candidates),
        }

    # ── Reranking ──
    rerank_start = time.perf_counter()
    sources = rerank_chunks(query, candidates, top_n=rerank_top_n)
    rerank_time = time.perf_counter() - rerank_start

    if settings.DEBUG:
        debug_info["reranking"] = {
            "input_chunks": len(candidates),
            "output_chunks": len(sources),
            "scores": [
                {"score": round(s["rerank_score"], 4), "source": s.get("metadata", {}).get("source")}
                for s in sources
            ],
        }

    # ── Build context (Block 3) ──
    context_texts = []
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata", {})
        source_name = meta.get("source", "Unknown")
        page = meta.get("page", "")
        context_texts.append(f"--- Chunk {i} ---\nSource: {source_name} {f'Page: {page}' if page else ''}\n{s['text']}\n")
    context = "\n".join(context_texts)

    # ── LLM generation ──
    llm_start = time.perf_counter()


    if stream:
        answer_gen = generate_streaming(query, context, backend_to_use)

        # For streaming, we add the turn after we know it's not NO_CONTEXT
        # (the streaming handler replaces the token inline)
        result = {
            "answer": answer_gen,
            "sources": sources,
            "llm_backend_used": backend_to_use,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
        if settings.DEBUG:
            result["debug"] = debug_info
        return result

    # Non-streaming path

    answer = generate_answer(query, context, backend_to_use)

    if NO_CONTEXT_TOKEN in answer.strip().upper():
        answer = "I could not find any relevant information in the provided documents to answer your question."
        sources = []

    llm_time = time.perf_counter() - llm_start
    total_time = time.perf_counter() - start_time
    total_time_ms = round(total_time * 1000, 2)

    logger.info(f"Query: '{query}' | Backend: {backend_to_use} | Candidates: {len(candidates)} | Reranked to: {len(sources)}")
    logger.info(
        f"Latency -> Embed: {embed_time*1000:.2f}ms, Retrieve: {retrieve_time*1000:.2f}ms, "
        f"Rerank: {rerank_time*1000:.2f}ms, LLM: {llm_time*1000:.2f}ms, Total: {total_time_ms}ms"
    )
    for s in sources:
        logger.info(
            f"Rerank Score: {s['rerank_score']:.4f} | Fusion Score: {s['similarity_score']:.4f} "
            f"| Source: {s.get('metadata', {}).get('source')}"
        )

    if settings.DEBUG:
        debug_info["latency"] = {
            "embed_ms": round(embed_time * 1000, 2),
            "retrieve_ms": round(retrieve_time * 1000, 2),
            "rerank_ms": round(rerank_time * 1000, 2),
            "llm_ms": round(llm_time * 1000, 2),
            "total_ms": total_time_ms,
        }

    result = {
        "answer": answer,
        "sources": sources,
        "llm_backend_used": backend_to_use,
        "latency_ms": total_time_ms,
    }
    if settings.DEBUG:
        result["debug"] = debug_info

    return result