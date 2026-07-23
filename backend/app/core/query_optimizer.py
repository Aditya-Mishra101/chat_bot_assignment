"""
Query optimization strategies for improving RAG retrieval quality.

Supports two strategies (toggleable via config):
  - Multi-Query: Rephrase the query 2-3 ways to improve recall
  - Decomposition: Split multi-part questions into sub-questions
"""

import asyncio
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_optimizer_llm

from app.core.config import settings

logger = logging.getLogger("rag_query_optimizer")


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
        "A question is multi-part ONLY if it explicitly asks about multiple distinct "
        "things (e.g. joined by 'and', containing multiple question marks, or asking "
        "about clearly separate topics). Requests like 'summarize X', 'tell me about X', "
        "'explain X', or any single question about one topic are SINGLE-PART, even if "
        "answering them requires covering multiple aspects (plot, theme, characters, etc.) "
        "— do NOT split these into multiple sub-questions.\n"
        "Only split the question into the parts EXPLICITLY asked by the user — "
        "do not invent, infer, or add any additional sub-questions the user did "
        "not ask.\n"
        "If the question is single-part, return it as-is (just one line, unchanged).\n"
        "Return ONLY the sub-questions, one per line, numbered 1-N.",
    ),
    ("human", "{query}"),
])


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


def _looks_multi_part(query: str) -> bool:
    """Heuristic pre-filter: only send a query to the decomposition LLM if it
    genuinely shows signs of asking multiple distinct things. Small local
    models unreliably follow prompt-only constraints against over-decomposing
    single-topic requests like 'summarize X' — this guard prevents wasted
    LLM calls AND prevents fabricated sub-questions from polluting retrieval."""
    q = query.lower().strip()

    single_part_signals = (
        q.startswith((
            "summarize", "summarise", "tell me about", "explain",
            "describe", "what is", "who is", "what are", "give me a summary",
        ))
    )
    multi_signals = (
        q.count("?") > 1
        or (" and " in q and "who" in q or " and " in q and "what" in q or " and " in q and "why" in q)
    )

    if single_part_signals and q.count("?") <= 1:
        return False

    return multi_signals


async def multi_query_expand_async(query: str) -> list[str]:
    llm = get_optimizer_llm()
    chain = _multi_query_prompt | llm | StrOutputParser()
    raw = await chain.ainvoke({"query": query})

    queries = _parse_numbered_lines(raw, max_items=3)

    if settings.DEBUG:
        logger.info(f"[DEBUG] Multi-query expansions: {queries}")

    return queries


async def decompose_query_async(query: str) -> list[str]:
    if not _looks_multi_part(query):
        if settings.DEBUG:
            logger.info(f"[DEBUG] Skipping decomposition — query doesn't look multi-part: {query}")
        return [query]

    llm = get_optimizer_llm()
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
      - "debug_info": dict of optimization steps performed
    """
    debug_info = {"original_query": query, "optimizations_applied": []}

    queries = [query]
    decomposed = False

    try:
        if settings.ENABLE_DECOMPOSITION and _looks_multi_part(query):
            sub_qs = await decompose_query_async(query)
            if len(sub_qs) > 1:
                queries = sub_qs
                decomposed = True
                debug_info["optimizations_applied"].append("decomposition")
                debug_info["sub_questions"] = sub_qs
        elif settings.ENABLE_MULTI_QUERY:
            expanded = await multi_query_expand_async(query)
            if expanded:
                queries.extend(expanded)
                debug_info["optimizations_applied"].append("multi_query")
                debug_info["expanded_queries"] = expanded
    except Exception as e:
        logger.warning(f"Query optimization failed: {e}")

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
        "decomposed": decomposed,
        "debug_info": debug_info,
    }