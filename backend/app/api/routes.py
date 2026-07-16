import time

from fastapi import APIRouter, HTTPException

from app.models.schemas import ChatRequest, ChatResponse, IngestResponse, RetrievedChunk
from app.core.retrieval import answer_query
from app.core.vector_store import collection_count
from app.core.config import settings

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "documents_indexed": collection_count(settings.COLLECTION_NAME),
    }


@router.post("/ingest", response_model=IngestResponse)
def ingest():
    from app.core.ingest import run_ingestion

    start = time.perf_counter()
    stats = run_ingestion()
    stats["duration_seconds"] = round(time.perf_counter() - start, 2)
    return IngestResponse(**stats)


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    result = answer_query(
        query=request.query,
        llm_backend_override=request.llm_backend,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=[
            RetrievedChunk(
                text=c["text"],
                metadata=c["metadata"],
                similarity_score=c["similarity_score"],
            )
            for c in result["sources"]
        ],
        llm_backend_used=result["llm_backend_used"],
        latency_ms=result["latency_ms"],
    )