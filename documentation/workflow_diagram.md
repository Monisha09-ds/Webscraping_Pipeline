# System Workflow Diagram

This diagram illustrates the detailed data flow through the Web Scraping and RAG pipeline.

```mermaid
flowchart TD
    %% Nodes
    subgraph Scraper ["🕷️ Web Scraper (Recursive / Frontier)"]
        direction TB
        Start([Start URL])
        Init[Initialize Frontier]
        
        subgraph Loop ["Crawl Loop"]
            Pop[Pop URL from Frontier\n(BFS: Queue / DFS: Stack)]
            CheckDepth{Depth > Max?}
            Crawl[Crawl Single Node]
            Fetch[Fetch Page Content]
            Save[Save as Markdown]
            ExtractLinks[Extract Links]
            Filter{Filter Links\n(Domain/Depth)}
            Push[Push to Frontier]
        end
        
        Disk[("📂 sitecontent/")]
    end

    subgraph Ingestion ["⚙️ Ingestion Pipeline"]
        Loader[Markdown Loader]
        Chunker[Text Chunker]
        Embedder[Local Embedder\n(SentenceTransformers)]
        VectorDB[("🗄️ ChromaDB")]
    end

    subgraph RAG ["🧠 RAG & Chat"]
        User([User Query])
        ChatUI[Streamlit UI]
        API[FastAPI Backend]
        Retriever[Context Retriever]
        Prompt[Prompt Builder]
        LLM[("🤖 LLM (Gemini)")]
        Answer([Final Answer])
    end

    %% Edges - Scraper Logic
    Start --> Init
    Init --> PushStart[Push Start URL]
    PushStart --> Pop
    
    Pop --> CheckDepth
    CheckDepth -- Yes --> Pop
    CheckDepth -- No --> Crawl
    
    Crawl --> Fetch
    Fetch --> Save
    Save --> Disk
    Save --> ExtractLinks
    
    ExtractLinks --> Filter
    Filter -- Accepted --> Push
    Push --> Pop
    Filter -- Rejected --> Pop

    %% Edges - Ingestion Flow
    Disk -->|Read .md| Loader
    Loader -->|Text| Chunker
    Chunker -->|Chunks| Embedder
    Embedder -->|Vectors| VectorDB

    %% Edges - RAG Flow
    User --> ChatUI
    ChatUI -->|POST /chat| API
    API -->|Query| Retriever
    Retriever -->|Search| VectorDB
    VectorDB -->|Top-k Chunks| Retriever
    Retriever -->|Context| Prompt
    Prompt -->|Prompt + Context| LLM
    LLM -->|Generated Text| Answer
    Answer --> ChatUI
```
