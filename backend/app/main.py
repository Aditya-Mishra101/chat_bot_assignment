import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.warmup import warmup_models

logger = logging.getLogger("rag_main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await asyncio.to_thread(warmup_models)
    except Exception as exc:
        logger.warning("Startup warmup failed (first request may be slower): %s", exc)
    yield


app = FastAPI(
    title="RAG Chatbot",
    description=(
        "Retrieval-Augmented Generation chatbot over a small document set "
        "(FastAPI + Qdrant + Ollama/API LLM toggle)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "RAG chatbot is running. See /docs for the API."}
