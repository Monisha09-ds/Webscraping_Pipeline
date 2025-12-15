# RAG & Chatbot Documentation

## Overview
The RAG (Retrieval-Augmented Generation) pipeline transforms the scraped content into a queryable knowledge base. It uses vector embeddings to retrieve relevant context and an LLM (Large Language Model) to generate accurate answers.

## Folder Structure
Relevant directories for RAG:

```text
src/
├── vectorstore/        # Database & Embeddings
│   ├── store.py        # ChromaDB wrapper
│   ├── embeddings.py   # SentenceTransformers wrapper
│   └── chunker.py      # Text splitting logic
├── rag/                # RAG Logic
│   ├── retrieve.py     # Context retrieval
│   ├── answer.py       # Prompt engineering & generation
│   └── llm.py          # LLM Interface (Gemini/Local)
├── pipelines/          # Workflows
│   ├── ingest.py       # Ingestion script
│   └── query.py        # CLI Chat script
└── frontend/           # User Interface
    └── chat_interface.py # Streamlit Chat UI
```

## System Design

### 1. Ingestion Pipeline (`src/pipelines/ingest.py`)
- **Input**: Markdown files from `sitecontent/`.
- **Chunking**: Splits text into manageable chunks (e.g., 450 tokens) with overlap.
- **Embedding**: Converts text chunks into vector representations using `sentence-transformers/all-MiniLM-L6-v2`.
- **Storage**: Saves vectors and metadata (URL, title) into **ChromaDB** (`chroma_store/`).

### 2. Retrieval (`src/rag/retrieve.py`)
- **Query Embedding**: Converts user query into a vector.
- **Search**: Performs a similarity search (Cosine Similarity) in ChromaDB to find the top-k most relevant chunks.

### 3. Generation (`src/rag/answer.py`)
- **Context Assembly**: Combines retrieved chunks into a context block.
- **Prompting**: Constructs a prompt instructing the LLM to answer based *only* on the provided context.
- **LLM**: Uses **Google Gemini 1.5 Flash** (via API) or a local model to generate the final response.

## Usage

### 1. Ingestion (Build the Knowledge Base)
Run this after scraping to populate the database.

```bash
python src/pipelines/ingest.py
```

### 2. Chat (CLI)
Test the RAG pipeline in the terminal.

```bash
python src/pipelines/query.py --mode api
```

### 3. Chat API (FastAPI)
The chat backend runs on port **5003**.

**Endpoints:**
- `POST /ingest`: Trigger ingestion for a specific folder.
- `POST /chat/init`: Initialize a chat session.
- `POST /chat`: Send a message and get a response.

**Start Server:**
```bash
python src/api/chat_api.py
```

### 4. Streamlit Chat UI
The primary user interface for chatting.

**Run:**
```bash
streamlit run src/frontend/chat_interface.py
```
