"""
Query optimization strategies for improving RAG retrieval quality.

Supports three strategies (all toggleable via config):
  - HyDE: Generate a hypothetical answer, embed it for better vector similarity
  - Multi-Query: Rephrase the query 2-3 ways to improve recall
  - Decomposition: Split multi-part questions into sub-questions
"""

import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

logger = logging.getLogger("rag_query_optimizer")


_hyde_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a document content predictor. Given a question, write a short "
        "paragraph (3-4 sentences) that a document might contain as the answer. "
        "Do NOT answer the question yourself — imagine what a relevant document passage would say. "
        "Be specific and factual in tone.",
    ),
    ("human", "{query}"),
])

_multi_query_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a search query expander. Given a user question, generate exactly "
        "3 alternative phrasings of the same question. Each rephrasing should use "
        "different wording but ask the same thing.\n"
        "Return ONLY the 3 rephrased queries, one per line, numbered 1-3. "
        "Do not include the original query.",
    ),
    ("human", "{query}"),
])

_decomposition_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a question decomposer. Given a complex or multi-part question, "
        "break it into simpler, independent sub-questions that can each be answered "
        "separately from a document corpus.\n"
        "If the question is already simple and single-part, return it as-is.\n"
        "Return ONLY the sub-questions, one per line, numbered 1-N.",
    ),
    ("human", "{query}"),
])


def _get_optimizer_llm():
    """Get a lightweight LLM instance for query optimization tasks.
    Plain sync function — get_llm just constructs a client object,
    it doesn't need to be async."""
    from app.core.llm import get_optimizer_llm
    return get_optimizer_llm()


def _parse_numbered_lines(raw: str, max_items: int | None = None) -> list[str]:
    items = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for i in range(1, 10):
            for sep in [".", ")", ":"]:
                prefix = f"{i}{sep}"
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
        if line:
            items.append(line)
    return items[:max_items] if max_items else items


async def hyde_expand_async(query: str) -> str:
    llm = _get_optimizer_llm()
    chain = _hyde_prompt | llm | StrOutputParser()
    hypothetical = await chain.ainvoke({"query": query})

    if settings.DEBUG:
        logger.info(f"[DEBUG] HyDE hypothetical passage:\n{hypothetical[:300]}")

    return hypothetical


async def multi_query_expand_async(query: str) -> list[str]:
    llm = _get_optimizer_llm()
    chain = _multi_query_prompt | llm | StrOutputParser()
    raw = await chain.ainvoke({"query": query})

    queries = _parse_numbered_lines(raw, max_items=3)

    if settings.DEBUG:
        logger.info(f"[DEBUG] Multi-query expansions: {queries}")

    return queries


async def decompose_query_async(query: str) -> list[str]:
    llm = _get_optimizer_llm()
    chain = _decomposition_prompt | llm | StrOutputParser()
    raw = await chain.ainvoke({"query": query})

    sub_questions = _parse_numbered_lines(raw)

    if settings.DEBUG:
        logger.info(f"[DEBUG] Decomposed sub-questions: {sub_questions}")

    return sub_questions if len(sub_questions) > 1 else [query]


async def optimize_query(query: str) -> dict:
    """
    Run all enabled query optimization strategies IN PARALLEL and return:
      - "queries": list of queries to run retrieval on
      - "hyde_text": hypothetical passage (or None)
      - "debug_info": dict of optimization steps performed
    """
    debug_info = {"original_query": query, "optimizations_applied": []}

    tasks = {}
    if settings.ENABLE_HYDE:
        tasks["hyde"] = hyde_expand_async(query)
    if settings.ENABLE_MULTI_QUERY:
        tasks["multi_query"] = multi_query_expand_async(query)
    if settings.ENABLE_DECOMPOSITION:
        tasks["decomposition"] = decompose_query_async(query)

    results = {}
    if tasks:
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        results = dict(zip(tasks.keys(), gathered))

    hyde_text = None
    queries = [query]

    if "hyde" in results:
        if isinstance(results["hyde"], Exception):
            logger.warning(f"HyDE expansion failed: {results['hyde']}")
        else:
            hyde_text = results["hyde"]
            debug_info["optimizations_applied"].append("hyde")
            debug_info["hyde_passage"] = hyde_text[:300]

    decomposed = False
    if "decomposition" in results:
        if isinstance(results["decomposition"], Exception):
            logger.warning(f"Query decomposition failed: {results['decomposition']}")
        else:
            sub_qs = results["decomposition"]
            if len(sub_qs) > 1:
                queries = sub_qs
                decomposed = True
                debug_info["optimizations_applied"].append("decomposition")
                debug_info["sub_questions"] = sub_qs

    # Only fold in multi-query rephrasings if decomposition didn't already
    # split the question — avoids inflating retrieval/rerank breadth.
    if "multi_query" in results and not decomposed:
        if isinstance(results["multi_query"], Exception):
            logger.warning(f"Multi-query expansion failed: {results['multi_query']}")
        else:
            expanded = results["multi_query"]
            queries.extend(expanded)
            debug_info["optimizations_applied"].append("multi_query")
            debug_info["expanded_queries"] = expanded

    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.strip().lower()
        if q_lower not in seen:
            seen.add(q_lower)
            unique_queries.append(q.strip())

    debug_info["final_queries"] = unique_queries

    if settings.DEBUG:
        logger.info(f"[DEBUG] Query optimization result: {debug_info}")

    return {
        "queries": unique_queries,
        "hyde_text": hyde_text,
        "debug_info": debug_info,
    }