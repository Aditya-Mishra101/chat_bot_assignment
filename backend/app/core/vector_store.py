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
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    always_ram=True
                )
            )
        )

def upsert_chunks(
    collection_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    sparse_embeddings: list = None,
):
    client = get_qdrant_client()
    import uuid

    points = []
    for i, (chunk, embedding, metadata) in enumerate(zip(chunks, embeddings, metadatas)):
        vector = {"dense": embedding}
        if sparse_embeddings is not None:
            sparse = sparse_embeddings[i]
            vector["sparse"] = models.SparseVector(
                indices=sparse.indices.tolist(),
                values=sparse.values.tolist(),
            )
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": chunk, **metadata}
            )
        )

    client.upsert(collection_name=collection_name, points=points)

def search_chunks(
    collection_name: str,
    query_embedding: list[float],
    top_k: int = settings.CHUNK,
    query_sparse_embedding=None,
) -> list[dict]:
    """
    If query_sparse_embedding is provided, runs hybrid search (dense + sparse)
    fused via Reciprocal Rank Fusion. Otherwise falls back to dense-only search
    (keeps this function backward compatible if called without sparse).
    """
    client = get_qdrant_client()

    if query_sparse_embedding is not None:
        sparse_vec = models.SparseVector(
            indices=query_sparse_embedding.indices.tolist(),
            values=query_sparse_embedding.values.tolist(),
        )
        response = client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_embedding,
                    using="dense",
                    limit=top_k * 4, 
                ),
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=top_k * 4,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
    else:
        response = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            using="dense",
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