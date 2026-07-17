from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    GOOGLE_API_KEY: str
    HF_TOKEN: str
    EMBEDDING_MODEL_ID: str
    QDRANT_PATH: str
    COLLECTION_NAME: str
    DEFAULT_LLM_BACKEND: Literal["api", "ollama"] = "ollama"
    API_MODEL_NAME: str
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL_NAME: str
    TEMPERATURE: float
    CHUNK: int

settings = Settings()