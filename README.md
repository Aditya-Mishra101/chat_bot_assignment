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
EMBEDDING_MODEL_ID=your_embedding_model
QDRANT_PATH=./qdrant_data
COLLECTION_NAME=your_qdrant_collection
DEFAULT_LLM_BACKEND=api/ollama (apikey/local)

API_MODEL_NAME=your_api_model
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=your_ollama_model
TEMPERATURE=0.2 (for most creative 1 , for factual 0)

CHUNK=5 (chunks retrieved and sent)
```

3. **Run the App**
   ```bash
   uvicorn app.main:app --reload
   ```

## Project Structure

```text
Assignment_Chat_Bot
│
├── backend
│   ├── app
│   │   ├── api
│   │   │   └── routes.py
│   │   ├── core
│   │   │   ├── config.py
│   │   │   ├── embedding.py
│   │   │   ├── ingest.py
│   │   │   ├── llm.py
│   │   │   ├── retrieval.py
│   │   │   └── vector_store.py
│   │   ├── models
│   │   ├── schemas.py
│   │   └── main.py
│   ├── docs/
│   ├── qdrant_data/
│   ├── .env
│   └── requirements.txt
│
├── frontend
│   ├── app
│   │   ├── favicon.ico
│   │   ├── globals.css
│   │   ├── layout.js
│   │   └── page.js
│   ├── AGENTS.md
│   ├── CLAUDE.md
│   ├── eslint.config.mjs
│   ├── jsconfig.json
│   ├── next.config.mjs
│   ├── package-lock.json
│   ├── package.json
│   ├── postcss.config.mjs
│   └── README.md
│
├── .gitignore
└── README.md
```

## API Endpoints
- `GET /health`: Check status and number of indexed documents.
- `POST /ingest`: Trigger the document ingestion pipeline from the `./docs` folder.
- `POST /chat`: Query the chatbot. Body: `{"query": "Your question", "llm_backend": "api" | "ollama"}`