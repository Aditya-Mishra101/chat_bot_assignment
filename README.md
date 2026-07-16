# RAG Chatbot

An Intelligent Retrieval-Augmented Generation (RAG) Chatbot built using FastAPI, Qdrant (local), and a configurable LLM backend supporting both an API model and an open-source Ollama model.

## Features
- **Document Ingestion**: Parses PDFs and Markdown files from `./docs`.
- **Semantic Chunking**: Uses embedding-based semantic chunking to split text at natural boundaries.
- **Embeddings**: Uses `BAAI/bge-base-en-v1.5` by default (configurable via `.env`) generated locally or via HuggingFace Inference provider.
- **Vector DB**: Qdrant running locally without Docker.
- **LLM Backends**: Toggles seamlessly between `gpt-4o-mini` via OpenAI API and `llama3.1:8b`/`qwen3:8b` via Ollama.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   HF_TOKEN=your_huggingface_token
   EMBEDDING_MODEL_ID=BAAI/bge-base-en-v1.5
   QDRANT_PATH=./qdrant_data
   COLLECTION_NAME=rag_docs
   DEFAULT_LLM_BACKEND=api
   API_MODEL_NAME=gpt-4o-mini
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL_NAME=llama3.1:8b
   ```

3. **Run the App**
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints
- `GET /health`: Check status and number of indexed documents.
- `POST /ingest`: Trigger the document ingestion pipeline from the `./docs` folder.
- `POST /chat`: Query the chatbot. Body: `{"query": "Your question", "llm_backend": "api" | "ollama"}`
