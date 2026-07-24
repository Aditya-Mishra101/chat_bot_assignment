import logging
from collections.abc import AsyncIterator

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from app.core.config import settings

logger = logging.getLogger("rag_llm")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise, concise document assistant.

Rules:
- Answer ONLY using the CONTEXT below. Do not use prior knowledge.
- If multiple context chunks are relevant, synthesize them into a coherent answer.
- If the user asks multiple distinct questions in one message, answer EVERY part separately.
  Use a clear heading or bullet for each sub-question. Do not skip any part the user asked.
- If you can answer some parts but not others, answer what you can and explicitly say which
  part lacks sufficient context. Do NOT respond with [[NO_CONTEXT]] when any part is answerable.
- Context may be grouped by sub-question — use the relevant section for each part.
- Cite sources inline after each key claim as [Source: filename]. Use the source name from the chunk metadata, without file extension or chunk numbers.
- Keep answers focused and well-structured. Use bullet points for lists.
- Respond with exactly [[NO_CONTEXT]] ONLY when NONE of the user's questions can be answered from the context.

Context:
{context}
"""
    ),
    ("human", "{query}")
])

NO_CONTEXT_TOKEN = "[[NO_CONTEXT]]"

_llm_cache: dict[str, ChatOllama | ChatGoogleGenerativeAI] = {}
_chain_cache: dict[str, Runnable] = {}


def get_llm(backend: str) -> ChatOllama | ChatGoogleGenerativeAI:
    if backend not in _llm_cache:
        if backend == "api":
            _llm_cache[backend] = ChatGoogleGenerativeAI(
                model=settings.API_MODEL_NAME,
                api_key=settings.GOOGLE_API_KEY,
                temperature=settings.TEMPERATURE,
            )
        elif backend == "ollama":
            _llm_cache[backend] = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL_NAME,
                temperature=settings.TEMPERATURE,
            )
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")
    return _llm_cache[backend]


def get_chain(backend: str) -> Runnable:
    if backend not in _chain_cache:
        _chain_cache[backend] = prompt | get_llm(backend) | StrOutputParser()
    return _chain_cache[backend]


async def generate_answer_async(query: str, context: str, backend: str) -> str:
    chain = get_chain(backend)
    return await chain.ainvoke({"context": context, "query": query})


def generate_answer(query: str, context: str, backend: str) -> str:
    chain = get_chain(backend)
    return chain.invoke({"context": context, "query": query})


async def generate_streaming_async(
    query: str, context: str, backend: str
) -> AsyncIterator[str]:
    chain = get_chain(backend)
    buffer = ""
    threshold = len(NO_CONTEXT_TOKEN) + 5
    released = False

    async for chunk in chain.astream({"context": context, "query": query}):
        if released:
            yield chunk
            continue

        buffer += chunk
        if len(buffer) < threshold:
            continue

        if NO_CONTEXT_TOKEN in buffer.upper():
            yield (
                "I could not find any relevant information in the provided "
                "documents to answer your question."
            )
            return

        released = True
        yield buffer

    if not released and buffer:
        if NO_CONTEXT_TOKEN in buffer.upper():
            yield (
                "I could not find any relevant information in the provided "
                "documents to answer your question."
            )
        else:
            yield buffer


def generate_streaming(query: str, context: str, backend: str):
    chain = get_chain(backend)
    gen = chain.stream({"context": context, "query": query})

    buffer = ""
    threshold = len(NO_CONTEXT_TOKEN) + 5
    released = False

    for chunk in gen:
        if released:
            yield chunk
            continue

        buffer += chunk
        if len(buffer) < threshold:
            continue

        if NO_CONTEXT_TOKEN in buffer.upper():
            yield (
                "I could not find any relevant information in the provided "
                "documents to answer your question."
            )
            return

        released = True
        yield buffer

    if not released and buffer:
        if NO_CONTEXT_TOKEN in buffer.upper():
            yield (
                "I could not find any relevant information in the provided "
                "documents to answer your question."
            )
        else:
            yield buffer


_optimizer_llm = None


def get_optimizer_llm():
    """Lightweight LLM dedicated to query optimization tasks (HyDE, multi-query,
    decomposition) — deliberately smaller/cheaper than the main generation
    backend, since these are simple rephrase/split tasks, not deep reasoning."""
    global _optimizer_llm
    if _optimizer_llm is None:
        if settings.DEFAULT_LLM_BACKEND == "api":
            _optimizer_llm = ChatGoogleGenerativeAI(
                model=settings.API_MODEL_NAME,
                api_key=settings.GOOGLE_API_KEY,
                temperature=0.0,
            )
        else:
            _optimizer_llm = ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=settings.OLLAMA_MODEL_NAME,
                temperature=0.0,
            )
    return _optimizer_llm
