# Project Overview: Web Scraping & RAG Pipeline

## Goal
To build a comprehensive system that scrapes website content, processes it into a knowledge base, and enables intelligent, context-aware Q&A through a chatbot interface.

## Architecture

```mermaid
graph TD
    A[Target Website] -->|Scraper| B(Markdown Files)
    B -->|Ingest Pipeline| C{Vector Store}
    C -->|ChromaDB| D[Knowledge Base]
    
    User -->|Query| E[Chat Interface]
    E -->|API| F[RAG Engine]
    F -->|Retrieve| D
    F -->|Generate| G[LLM (Gemini)]
    G -->|Answer| E
```

## Technology Stack
- **Language**: Python 3.10+
- **Scraping**: Playwright, Crawl4AI
- **Database**: ChromaDB (Vector Store)
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
- **LLM**: Google Gemini 1.5 Flash (via `google-genai`)
- **API**: FastAPI
- **Frontend**: Streamlit

## Workflow
1.  **Scrape**: The **Scraper** traverses a website (BFS/DFS) and saves pages as Markdown in `sitecontent/`.
2.  **Ingest**: The **Ingestion Pipeline** reads Markdown files, chunks them, generates embeddings, and stores them in **ChromaDB**.
3.  **Query**: The **Chatbot** takes a user question, retrieves relevant chunks from ChromaDB, and uses the **LLM** to generate an answer based on that context.

## Documentation Index
- [Scraper Documentation](./scraper.md) - Details on the crawling engine.
- [RAG & Chatbot Documentation](./rag_chatbot.md) - Details on the vector store and Q&A pipeline.
