import time
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from app.core.config import settings
from app.core.embedding import get_embeddings, get_embedding_model, get_sparse_embeddings
from app.core.vector_store import delete_collection, ensure_collection, upsert_chunks

def load_documents(docs_dir: Path):
    docs = []
    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        return docs
    
    for file_path in docs_dir.rglob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() == '.pdf':
                loader = PyMuPDFLoader(str(file_path))
                docs.extend(loader.load())
            elif file_path.suffix.lower() in ['.md', '.txt']:
                loader = TextLoader(str(file_path))
                docs.extend(loader.load())
    return docs

def run_ingestion() -> dict:
    start_time = time.perf_counter()
    docs_dir = Path("./docs")
    
    print("[1/5] Loading documents...")
    docs = load_documents(docs_dir)
    if not docs:
        print("No documents found in ./docs")
        return {"status": "success", "documents_processed": 0, "chunks_created": 0}

    print("[2/5] Chunking documents (Semantic Chunking)...")
    text_splitter = SemanticChunker(
        get_embedding_model(),
        buffer_size=2,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,
        min_chunk_size=200
    )
    chunks = text_splitter.split_documents(docs)
    
    print("[3/5] Generating dense embeddings...")
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    dense_embeddings = get_embeddings(texts)

    print("[4/5] Generating sparse (BM25) embeddings...")
    sparse_embeddings = get_sparse_embeddings(texts)
    
    print(f"[5/5] Upserting to Qdrant collection: {settings.COLLECTION_NAME}")
    delete_collection(settings.COLLECTION_NAME)
    ensure_collection(settings.COLLECTION_NAME)
    upsert_chunks(settings.COLLECTION_NAME, texts, dense_embeddings, metadatas, sparse_embeddings)
    
    duration = time.perf_counter() - start_time
    print(f"Ingestion completed in {duration:.2f}s")
    
    return {
        "status": "success",
        "documents_processed": len(set(doc.metadata.get("source", "") for doc in docs)),
        "chunks_created": len(chunks),
        "duration_seconds": round(duration, 2)
    }

if __name__ == "__main__":
    run_ingestion()