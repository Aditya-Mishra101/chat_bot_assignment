import asyncio
import time
import logging

from app.core.config import settings
from app.core.embedding import get_embeddings, get_sparse_embeddings
from app.core.vector_store import search_chunks
from app.core.reranker import rerank_chunks
from app.core.llm import (
    NO_CONTEXT_TOKEN,
    generate_answer_async,
    generate_streaming_async,
)
from app.core.query_optimizer import optimize_query, _looks_multi_part

logger = logging.getLogger("rag_retrieval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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


async def _embed_queries(queries: list[str]) -> tuple[list[list[float]], list]:
    dense_embs, sparse_embs = await asyncio.gather(
        asyncio.to_thread(get_embeddings, queries),
        asyncio.to_thread(get_sparse_embeddings, queries),
    )
    return dense_embs, sparse_embs


async def _search_all_queries(
    queries: list[str],
    dense_embs: list[list[float]],
    sparse_embs: list,
    retrieve_k: int,
) -> dict[str, list[dict]]:
    async def _search_one(
        q: str, emb: list[float], sparse_emb
    ) -> tuple[str, list[dict]]:
        candidates = await asyncio.to_thread(
            search_chunks,
            settings.COLLECTION_NAME,
            emb,
            retrieve_k,
            sparse_emb,
        )
        return q, candidates

    results = await asyncio.gather(
        *[
            _search_one(q, emb, sparse_emb)
            for q, emb, sparse_emb in zip(queries, dense_embs, sparse_embs)
        ]
    )
    return dict(results)


async def _retrieve_for_queries(
    queries: list[str], retrieve_k: int
) -> dict[str, list[dict]]:
    dense_embs, sparse_embs = await _embed_queries(queries)
    return await _search_all_queries(queries, dense_embs, sparse_embs, retrieve_k)


async def _run_retrieval_with_optimization(
    query: str, retrieve_k: int
) -> tuple[dict, dict[str, list[dict]], float, float]:
    """
    Overlap query optimization with initial retrieval when safe.

    For single-part queries, retrieval on the original query runs in parallel
    with multi-query expansion. Decomposition still runs first when the query
    looks multi-part, since sub-questions must be known before searching.
    """
    might_decompose = settings.ENABLE_DECOMPOSITION and _looks_multi_part(query)

    if might_decompose:
        opt_start = time.perf_counter()
        opt_result = await optimize_query(query)
        opt_time = time.perf_counter() - opt_start

        retrieval_start = time.perf_counter()
        grouped_candidates = await _retrieve_for_queries(
            opt_result["queries"], retrieve_k
        )
        post_opt_retrieval_time = time.perf_counter() - retrieval_start
        return (
            opt_result,
            grouped_candidates,
            opt_time,
            opt_time + post_opt_retrieval_time,
        )

    opt_start = time.perf_counter()
    opt_task = asyncio.create_task(optimize_query(query))
    initial_task = asyncio.create_task(_retrieve_for_queries([query], retrieve_k))

    opt_result, initial_grouped = await asyncio.gather(opt_task, initial_task)
    opt_time = time.perf_counter() - opt_start

    queries_to_run = opt_result["queries"]
    original_key = query.strip().lower()
    additional_queries = [
        q for q in queries_to_run if q.strip().lower() != original_key
    ]

    grouped_candidates = dict(initial_grouped)
    retrieve_start = time.perf_counter()

    if additional_queries:
        extra_grouped = await _retrieve_for_queries(additional_queries, retrieve_k)
        grouped_candidates.update(extra_grouped)

    retrieve_time = time.perf_counter() - retrieve_start
    embed_time = opt_time + retrieve_time

    return opt_result, grouped_candidates, opt_time, embed_time


async def _rerank_sources(
    query: str,
    queries_to_run: list[str],
    grouped_candidates: dict[str, list[dict]],
    is_decomposed: bool,
    rerank_top_n: int,
) -> list[dict]:
    if is_decomposed:
        per_question_top_n = max(1, rerank_top_n // len(queries_to_run))

        async def _rerank_one(q: str) -> list[dict]:
            candidates_for_q = _deduplicate_chunks(grouped_candidates.get(q, []))
            return await asyncio.to_thread(
                rerank_chunks, q, candidates_for_q, per_question_top_n
            )

        rerank_results = await asyncio.gather(
            *[_rerank_one(q) for q in queries_to_run]
        )

        merged = []
        seen_texts = set()
        for top_for_q in rerank_results:
            for chunk in top_for_q:
                if chunk["text"] not in seen_texts:
                    seen_texts.add(chunk["text"])
                    merged.append(chunk)
        return merged

    all_candidates_flat = []
    for candidates in grouped_candidates.values():
        all_candidates_flat.extend(candidates)
    candidates = _deduplicate_chunks(all_candidates_flat)
    return await asyncio.to_thread(rerank_chunks, query, candidates, rerank_top_n)


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

    opt_result, grouped_candidates, opt_time, embed_retrieve_time = (
        await _run_retrieval_with_optimization(query, retrieve_k)
    )

    queries_to_run = opt_result["queries"]
    is_decomposed = opt_result["decomposed"]

    if settings.DEBUG:
        debug_info["query_optimization"] = opt_result["debug_info"]
        debug_info["query_optimization"]["latency_ms"] = round(opt_time * 1000, 2)

    all_candidates_flat = []
    for candidates in grouped_candidates.values():
        all_candidates_flat.extend(candidates)

    if settings.DEBUG:
        debug_info["retrieval"] = {
            "queries_run": len(queries_to_run),
            "total_candidates_before_dedup": len(all_candidates_flat),
            "candidates_after_dedup": len(_deduplicate_chunks(all_candidates_flat)),
        }

    rerank_start = time.perf_counter()
    sources = await _rerank_sources(
        query, queries_to_run, grouped_candidates, is_decomposed, rerank_top_n
    )
    rerank_time = time.perf_counter() - rerank_start

    if settings.DEBUG:
        if is_decomposed:
            input_chunks = sum(
                len(_deduplicate_chunks(grouped_candidates.get(q, [])))
                for q in queries_to_run
            )
        else:
            input_chunks = len(_deduplicate_chunks(all_candidates_flat))

        debug_info["reranking"] = {
            "input_chunks": input_chunks,
            "output_chunks": len(sources),
            "strategy": "per_subquestion" if is_decomposed else "pooled",
            "scores": [
                {
                    "score": round(s.get("rerank_score", 0), 4),
                    "source": s.get("metadata", {}).get("source"),
                }
                for s in sources
            ],
        }

    context_texts = []
    for i, s in enumerate(sources, 1):
        meta = s.get("metadata", {})
        source_name = meta.get("source", "Unknown")
        page = meta.get("page", "")
        context_texts.append(
            f"--- Chunk {i} ---\nSource: {source_name} "
            f"{f'Page: {page}' if page else ''}\n{s['text']}\n"
        )
    context = "\n".join(context_texts)

    llm_start = time.perf_counter()

    if stream:
        answer_gen = generate_streaming_async(query, context, backend_to_use)
        result = {
            "answer": answer_gen,
            "sources": sources,
            "llm_backend_used": backend_to_use,
            "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        }
        if settings.DEBUG:
            result["debug"] = debug_info
        return result

    answer = await generate_answer_async(query, context, backend_to_use)

    if NO_CONTEXT_TOKEN in answer.strip().upper():
        answer = (
            "I could not find any relevant information in the provided "
            "documents to answer your question."
        )
        sources = []

    llm_time = time.perf_counter() - llm_start
    total_time = time.perf_counter() - start_time
    total_time_ms = round(total_time * 1000, 2)

    logger.info(
        f"Query: '{query}' | Backend: {backend_to_use} | Reranked to: {len(sources)}"
    )
    logger.info(
        f"Latency -> Embed+Retrieve: {embed_retrieve_time*1000:.2f}ms, "
        f"Rerank: {rerank_time*1000:.2f}ms, LLM: {llm_time*1000:.2f}ms, "
        f"Total: {total_time_ms}ms"
    )
    for s in sources:
        logger.info(
            f"Rerank Score: {s.get('rerank_score', 0):.4f} | "
            f"Source: {s.get('metadata', {}).get('source')}"
        )

    if settings.DEBUG:
        debug_info["latency"] = {
            "embed_retrieve_ms": round(embed_retrieve_time * 1000, 2),
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
