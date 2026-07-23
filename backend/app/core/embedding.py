from langchain_huggingface import HuggingFaceEndpointEmbeddings
from fastembed import SparseTextEmbedding
from app.core.config import settings

_embedding_model = None
_sparse_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:    
        _embedding_model = HuggingFaceEndpointEmbeddings(
            model=settings.EMBEDDING_MODEL_ID,
            huggingfacehub_api_token=settings.HF_TOKEN,
        )
    return _embedding_model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.embed_documents(texts)

def get_query_embedding(query: str) -> list[float]:
    model = get_embedding_model()
    return model.embed_query(query)


def get_sparse_model() -> SparseTextEmbedding:
    """Loaded once and reused — this must be the SAME model/instance type
    used at both ingestion and query time, or sparse vector vocabularies
    won't align and hybrid search will silently return garbage."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_EMBEDDING_MODEL)
    return _sparse_model

def get_sparse_embeddings(texts: list[str]):
    """Used at ingestion time — returns a list of SparseEmbedding objects."""
    model = get_sparse_model()
    return list(model.embed(texts))

def get_query_sparse_embedding(query: str):
    """Used at query time — returns a single SparseEmbedding object."""
    model = get_sparse_model()
    return list(model.embed([query]))[0]