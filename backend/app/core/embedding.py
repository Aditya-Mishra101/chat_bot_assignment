from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app.core.config import settings

_embedding_model = None

def get_embedding_model():
    return HuggingFaceEndpointEmbeddings(
        model=settings.EMBEDDING_MODEL_ID,
        huggingfacehub_api_token=settings.HF_TOKEN,
    )

def get_embeddings(texts: list[str]) -> list[list[float]]:

    model = get_embedding_model()
    return model.embed_documents(texts)

def get_query_embedding(query: str) -> list[float]:

    model = get_embedding_model()
    return model.embed_query(query)
