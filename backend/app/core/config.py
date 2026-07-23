from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GOOGLE_API_KEY:str
    HF_TOKEN:str
    EMBEDDING_MODEL_ID:str
    QDRANT_PATH:str
    COLLECTION_NAME:str
    DEFAULT_LLM_BACKEND: Literal["api", "ollama"] = "ollama"
    API_MODEL_NAME:str
    OLLAMA_BASE_URL:str
    OLLAMA_MODEL_NAME:str
    SPARSE_EMBEDDING_MODEL: str
    TEMPERATURE:float
    CHUNK:int
    # ── Retrieval tuning ──
    RE_RANKER_MODEL:str
    RETRIEVE_K:int
    RERANK_TOP_N:int
    RERANK_SCORE_THRESHOLD:float
    # ── Query optimization feature flags ──
    ENABLE_MULTI_QUERY:bool
    ENABLE_DECOMPOSITION:bool
    # ── Debug ──
    DEBUG:bool

settings = Settings()