# 🕷️ Web Scraper & RAG Pipeline

A powerful, full-stack system that scrapes website content, builds a vector knowledge base, and enables intelligent, context-aware Q&A via a Chatbot.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4-orange)
![Gemini](https://img.shields.io/badge/LLM-Gemini-purple)

## 📖 Documentation
- **[Project Overview](documentation/project_overview.md)**: High-level architecture and design.
- **[Scraper Guide](documentation/scraper.md)**: Detailed crawler logic, BFS/DFS strategies, and parameters.
- **[RAG & Chatbot](documentation/rag_chatbot.md)**: How the retrieval and generation pipeline works.
- **[Workflow Diagram](documentation/workflow_diagram.md)**: Visual data flow.

## 🚀 Features

### 1. Advanced Web Scraper
- **Recursive Crawling**: BFS (Breadth-First) or DFS (Depth-First) traversal.
- **Smart Extraction**: Converts HTML to clean **Markdown**.
- **Frontier Management**: Handles deduplication, depth limits, and domain filtering.
- **Resumable**: Saves progress to `sitecontent/` folder structure.

### 2. RAG Pipeline (Retrieval-Augmented Generation)
- **Ingestion**: Chunks markdown files and generates vector embeddings.
- **Vector Store**: Uses **ChromaDB** for efficient similarity search.
- **Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (local & fast).

### 3. Chatbot Interface
- **Context-Aware**: Answers questions based *only* on the scraped content.
- **Multi-Modal**: Supports CLI, API, and Streamlit Web UI.
- **LLM Support**: Integrated with **Google Gemini 1.5 Flash** (API) or local models.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo-url>
    cd Webscraping_Pipeline
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/Mac
    # .venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Environment Variables**:
    Copy `.env.example` to `.env` and add your keys.
    ```bash
    cp .env.example .env
    ```
    *   `GOOGLE_API_KEY`: Required for Gemini LLM.

## 🏃 Usage

### 1. Web Scraper
**Via CLI:**
```bash
# Scrape a website (BFS, depth 2)
python src/main.py --url https://example.com --strategy bfs --max-depth 2
```

**Via Streamlit UI:**
```bash
streamlit run src/frontend/scraper_interface.py
```

### 2. Ingest Data (Build Knowledge Base)
After scraping, process the data into ChromaDB:
```bash
python src/pipelines/ingest.py
```

### 3. Chatbot
**Via CLI:**
```bash
python src/pipelines/query.py --mode api
```

**Via Streamlit UI:**
```bash
streamlit run src/frontend/chat_interface.py
```

### 4. API Server
Run the backend APIs for integration:
```bash
# Chat API (Port 5003)
python src/api/chat_api.py

# Scraper API (Port 5005)
python src/api/scraper_endpoints.py
```

## 🧪 Testing
Run the test suite to verify all components:
```bash
pytest tests/
```

## 📂 Directory Structure
```text
src/
├── scraper/        # Crawling logic (Frontier, Fetcher, Saver)
├── vectorstore/    # ChromaDB & Embedding logic
├── rag/            # Retrieval & Answer generation
├── pipelines/      # Scripts (ingest.py, query.py)
├── api/            # FastAPI endpoints
├── frontend/       # Streamlit UIs
└── utils/          # Config & Helpers
tests/              # Unit & Integration tests
sitecontent/        # Scraped Markdown files (Output)
chroma_store/       # Vector Database (Output)
documentation/      # Detailed docs
```
