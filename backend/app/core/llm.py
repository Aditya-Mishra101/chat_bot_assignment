from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

def get_llm(backend: str):
    if backend == "api":
        return ChatOpenAI(
            model=settings.API_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2
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
    
    prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions using ONLY the provided context.

        Rules:
        - Answer strictly using the CONTEXT below — never use outside knowledge.
        - If the context has enough information, answer concisely and cite the source label(s) you used (e.g. "[Source 2]").
        - If multiple context sections are relevant, synthesize across them and cite all sources used.
        - If the context does not contain enough information, return exactly: NO
        - If the question is unrelated to the context, return exactly: NO
        - If multiple context sections are relevant, combine them.
        - Keep answers concise and factual.

        Context:
        {context}
        """
            ),
            ("human", "{query}")
        ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "context": context,
        "query": query
    })
