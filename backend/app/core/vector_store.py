from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.core.config import settings

_client = None

def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=settings.QDRANT_PATH)
    return _client

def delete_collection(collection_name: str):
    client = get_qdrant_client()

    collections = client.get_collections().collections

    if any(c.name == collection_name for c in collections):
        client.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")

def ensure_collection(collection_name: str, vector_size: int = 768):
    client = get_qdrant_client()
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True
                )
            )
        )

def upsert_chunks(collection_name: str, chunks: list[str], embeddings: list[list[float]], metadatas: list[dict]):
    client = get_qdrant_client()
    
    import uuid
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, **metadata}
        )
        for chunk, embedding, metadata in zip(chunks, embeddings, metadatas)
    ]
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )

def search_chunks(
    collection_name: str,
    query_embedding: list[float],
    top_k: int = settings.CHUNK,
) -> list[dict]:
    client = get_qdrant_client()

    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "text": point.payload.get("text", ""),
            "metadata": {k: v for k, v in point.payload.items() if k != "text"},
            "similarity_score": point.score,
        }
        for point in response.points
    ]

def collection_count(collection_name: str) -> int:
    client = get_qdrant_client()
    try:
        return client.count(collection_name=collection_name).count
    except Exception:
        return 0
