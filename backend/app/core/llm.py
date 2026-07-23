import logging
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

logger = logging.getLogger("rag_llm")

# ── Standard RAG prompt (no memory) ──
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise, concise document assistant.

Rules:
- Answer ONLY using the CONTEXT below. Do not use prior knowledge.
- If multiple context chunks are relevant, synthesize them into a coherent answer.
- Cite sources inline after each key claim as [Source: filename]. Use the source name from the chunk metadata, without file extension or chunk numbers.
- Keep answers focused and well-structured. Use bullet points for lists.
- If the context does not contain enough information to answer, respond with exactly: [[NO_CONTEXT]]
- If the question is unrelated to all provided context, respond with exactly: [[NO_CONTEXT]]

Context:
{context}
"""
    ),
    ("human", "{query}")
])

NO_CONTEXT_TOKEN = "[[NO_CONTEXT]]"

def get_llm(backend: str):
    if backend == "api":
        return ChatGoogleGenerativeAI(
            model=settings.API_MODEL_NAME,
            api_key=settings.GOOGLE_API_KEY,
            temperature=settings.TEMPERATURE
        )
    elif backend == "ollama":
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL_NAME,
            temperature=settings.TEMPERATURE
        )
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")

def generate_answer(query: str, context: str, backend: str) -> str:
    llm = get_llm(backend)
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "context": context,
        "query": query
    })

def generate_streaming(query: str, context: str, backend: str):
    llm = get_llm(backend)
    chain = prompt | llm | StrOutputParser()
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
            yield "I could not find any relevant information in the provided documents to answer your question."
            return

        released = True
        yield buffer
        
    if not released and buffer:
        if NO_CONTEXT_TOKEN in buffer.upper():
            yield "I could not find any relevant information in the provided documents to answer your question."
        else:
            yield buffer


_optimizer_llm = None

def get_optimizer_llm():
    global _optimizer_llm
    if _optimizer_llm is None:
        endpoint = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-1.5B-Instruct",
            provider="featherless-ai",
            huggingfacehub_api_token=settings.HF_TOKEN,
            max_new_tokens=256,
            temperature=0.2,
        )
        _optimizer_llm = ChatHuggingFace(llm=endpoint)
    return _optimizer_llm
