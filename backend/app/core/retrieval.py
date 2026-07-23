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

    retrieve_k = retrieve_k or settings.RETRIEVE_K
    rerank_top_n = rerank_top_n or settings.RERANK_TOP_N
    backend_to_use = llm_backend_override or settings.DEFAULT_LLM_BACKEND

    opt_start = time.perf_counter()
    opt_result = await optimize_query(query)
    opt_time = time.perf_counter() - opt_start

    queries_to_run = opt_result["queries"]
    is_decomposed = opt_result["decomposed"]

    if settings.DEBUG:
        debug_info["query_optimization"] = opt_result["debug_info"]
        debug_info["query_optimization"]["latency_ms"] = round(opt_time * 1000, 2)

    # ── Embedding + Retrieval, tracked PER QUERY (not flattened) ──
    embed_start = time.perf_counter()

    from app.core.embedding import get_embeddings, get_sparse_embeddings
    dense_embs = get_embeddings(queries_to_run)
    sparse_embs = get_sparse_embeddings(queries_to_run)

    per_query_candidates: dict[str, list[dict]] = {}

    for q, emb, sparse_emb in zip(queries_to_run, dense_embs, sparse_embs):
        per_query_candidates[q] = (emb, sparse_emb)

    embed_time = time.perf_counter() - embed_start

    retrieve_start = time.perf_counter()

    grouped_candidates: dict[str, list[dict]] = {}
    all_candidates_flat = []

    for q, (emb, sparse_emb) in per_query_candidates.items():
        candidates = search_chunks(
            collection_name=settings.COLLECTION_NAME,
            query_embedding=emb,
            top_k=retrieve_k,
            query_sparse_embedding=sparse_emb,
        )
        grouped_candidates[q] = candidates
        all_candidates_flat.extend(candidates)

    retrieve_time = time.perf_counter() - retrieve_start

    if settings.DEBUG:
        debug_info["retrieval"] = {
            "queries_run": len(queries_to_run),
            "total_candidates_before_dedup": len(all_candidates_flat),
            "candidates_after_dedup": len(_deduplicate_chunks(all_candidates_flat)),
        }

    # ── Reranking ──
    # Decomposition = different topics → rerank each sub-question separately
    # Multi-query / single = same topic → pool all candidates, rerank once
    rerank_start = time.perf_counter()

    if is_decomposed:
        per_question_top_n = max(1, rerank_top_n // len(queries_to_run))
        merged = []
        seen_texts = set()
        for q in queries_to_run:
            candidates_for_q = _deduplicate_chunks(grouped_candidates.get(q, []))
            top_for_q = rerank_chunks(q, candidates_for_q, top_n=per_question_top_n)
            for chunk in top_for_q:
                if chunk["text"] not in seen_texts:
                    seen_texts.add(chunk["text"])
                    merged.append(chunk)
        sources = merged
    else:
        candidates = _deduplicate_chunks(all_candidates_flat)
        sources = rerank_chunks(query, candidates, top_n=rerank_top_n)

    rerank_time = time.perf_counter() - rerank_start

    if settings.DEBUG:
        debug_info["reranking"] = {
            "input_chunks": len(candidates) if not is_decomposed else sum(len(_deduplicate_chunks(grouped_candidates.get(q, []))) for q in queries_to_run),
            "output_chunks": len(sources),
            "strategy": "per_subquestion" if is_decomposed else "pooled",
            "scores": [
                {"score": round(s.get("rerank_score", 0), 4), "source": s.get("metadata", {}).get("source")}
                for s in sources
            ],
        }

    context_texts = []
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata", {})
        source_name = meta.get("source", "Unknown")
        page = meta.get("page", "")
        context_texts.append(f"--- Chunk {i} ---\nSource: {source_name} {f'Page: {page}' if page else ''}\n{s['text']}\n")
    context = "\n".join(context_texts)

    llm_start = time.perf_counter()

    if stream:
        answer_gen = generate_streaming(query, context, backend_to_use)
        result = {
            "answer": answer_gen,
            "sources": sources,
            "llm_backend_used": backend_to_use,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
        if settings.DEBUG:
            result["debug"] = debug_info
        return result

    answer = generate_answer(query, context, backend_to_use)

    if NO_CONTEXT_TOKEN in answer.strip().upper():
        answer = "I could not find any relevant information in the provided documents to answer your question."
        sources = []

    llm_time = time.perf_counter() - llm_start
    total_time = time.perf_counter() - start_time
    total_time_ms = round(total_time * 1000, 2)

    logger.info(f"Query: '{query}' | Backend: {backend_to_use} | Reranked to: {len(sources)}")
    logger.info(
        f"Latency -> Embed: {embed_time*1000:.2f}ms, Retrieve: {retrieve_time*1000:.2f}ms, "
        f"Rerank: {rerank_time*1000:.2f}ms, LLM: {llm_time*1000:.2f}ms, Total: {total_time_ms}ms"
    )
    for s in sources:
        logger.info(
            f"Rerank Score: {s.get('rerank_score', 0):.4f} | Source: {s.get('metadata', {}).get('source')}"
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