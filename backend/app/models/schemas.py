from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class RetrievedChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    similarity_score: float

class ChatRequest(BaseModel):
    query: str = Field(..., description="User query")
    llm_backend: Optional[Literal["api", "ollama"]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[RetrievedChunk]
    llm_backend_used: str
    latency_ms: float
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug info (only present when DEBUG=true)")

class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    duration_seconds: float
