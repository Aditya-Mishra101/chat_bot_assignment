from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        model_kwargs = {'device': 'cpu'} 
        encode_kwargs = {'normalize_embeddings': True}
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_ID,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    return _embedding_model

def get_embeddings(texts: list[str]) -> list[list[float]]:

    model = get_embedding_model()
    return model.embed_documents(texts)

def get_query_embedding(query: str) -> list[float]:

    model = get_embedding_model()
    return model.embed_query(query)
