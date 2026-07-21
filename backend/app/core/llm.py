from typing import Iterator
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions using the provided context.

        Rules:
        - Answer strictly using the CONTEXT below.
        - If multiple context sections are relevant, combine them.
        - While specifying sources mention the source name and file name without extension, example: Source[source_name]. Don't mention chunk number or chunk as a source.
        - If the context does not contain enough information, respond with exactly this token and nothing else: [[NO_CONTEXT]]
        - If the question is unrelated to the context, respond with exactly this token and nothing else: [[NO_CONTEXT]]

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
