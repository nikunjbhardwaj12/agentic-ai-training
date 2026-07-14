# Task 03: Plan-and-Execute Agentic RAG API

A production-grade, stateful Multi-Agent system built using **LangGraph**, **FastAPI**, and **ChromaDB**. The system implements a dynamic **Plan-and-Execute** workflow with parallel research execution (Map-Reduce), a local RAG pipeline, and iterative self-evaluation loops to guarantee factuality and relevance.

---

## System Architecture

This project models agentic workflows as a directed, cyclical state graph, optimizing complex research tasks into coordinate steps:

```text
               +------------------+
               |   Planner Node   | <---------------+
               +------------------+                 |
                        |                           |
                        | (Generates N Sub-tasks)   |
                        v                           |
             +----------------------+               |
             |  Parallel Fan-Out    |               |
             +----------------------+               |
             /          |           \               |
            v           v            v              |
       [Researcher] [Researcher] [Researcher]       | (If Faithfulness/
            \           |            /              |  Relevance < 5)
             v          v           v               |
             +----------------------+               |
             |  Aggregation/Fan-In  |               |
             +----------------------+               |
                        |                           |
                        v                           |
               +------------------+                 |
               |   Writer Node    |                 |
               +------------------+                 |
                        |                           |
                        v                           |
               +------------------+                 |
               |  Evaluator Node  | ----------------+
               +------------------+
                        |
                        | (Approved: 5/5)
                        v
               +------------------+
               |   Final Answer   |
               +------------------+
```
1. **Planner Node:** Deconstructs a complex user query into distinct, parallelizable sub-queries.
2. **Researcher Node (Parallel Fan-Out):** Executes parallel database retrieval tasks against a local ChromaDB vector store.
3. **Aggregator Node (Fan-In):** Merges parallel outputs and deduplicates retrieved context utilizing custom reducers to maintain state.
4. **Writer Node:** Synthesizes the aggregated research findings into a structured draft.
5. **Evaluator Node:** Evaluates the draft on Faithfulness and Relevance (scale of 1-5). If criteria are not met, feedback is routed back to the Planner for recursive refinement.

---

## Key Solved Engineering Challenges
1. Multi-Threaded SQLite Concurrency Lock (ChromaDB)  
- Problem: During the Map-Reduce/Fan-out phase, multiple parallel Researcher threads attempt to query the Chroma vector store simultaneously. This resulted in an sqlite3.IntegrityError / UniqueConstraintError when parallel threads concurrently attempted to verify or initialize the collection schema.  

- Solution: Implemented a thread-safe synchronization lock (threading.Lock()) around the Chroma client initialization block within rag/retriever.py, guaranteeing serialized access to the database metadata layout without sacrificing query retrieval performance.

2. Windows-Specific Compilation Fallback  
- Problem: Local builds of ChromaDB heavily depend on C++ compilation binaries (hnswlib) which often fail to compile out-of-the-box on standard Windows environments.  

- Solution: Bypassed compilation errors by overriding default system settings with chromadb.api.segment.SegmentAPI and safely packaging dependencies to run on standard Python 3.10 virtual environments without requiring global C++ compiler workloads.

---

## Folder Structure

```text
task-03-plan-execute-api/
├── agents/                 # Modular stategraph nodes
│   ├── __init__.py         # Package identifier (empty)
│   ├── aggregator.py       # Thread-safe context consolidation
│   ├── evaluator.py        # Faithfulness and Relevance LLM validator
│   ├── planner.py          # Query decomposer
│   ├── researcher.py       # Parallel data-gatherer
│   └── writer.py           # Final synthesizer
├── api/                    # Web Application Layer
│   ├── schemas.py          # Request/Response Pydantic schemas
│   └── server.py           # FastAPI server with Server-Sent Events (SSE)
├── data/                   # Physical Storage
│   ├── chroma_db/          # Persistent local vector store
│   ├── documents/          # Raw reference documents for RAG
│   └── checkpoints.db      # SQLite database for LangGraph state checkpointers
├── rag/                    # Retrieval-Augmented Generation Logic
│   ├── __init__.py         # Package identifier (empty)
│   ├── ingest.py           # Document chunker & embedder
│   └── retriever.py        # Thread-safe vector space retriever
├── .env                    # System Environment Variables (Ignored in Git)
├── graph.py                # LangGraph StateGraph design and compilation
├── main.py                 # CLI Interface execution
├── state.py                # Global graph state definitions
└── requirements.txt        # Virtual environment dependencies
```
---
## Getting Started
1. **Environment Setup**  
Clone the repository and set up a Python virtual environment:
```text
PowerShell

# Create Virtual Environment
python -m venv .venv

# Activate Virtual Environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```
2. **Environment Variables Configuration**   
Create a .env file in the root directory based on .env.example:  
```text
Code snippet

OPENAI_API_KEY=your_openai_key_here
# If using HuggingFace models locally:
HF_TOKEN=your_huggingface_token_here
```
3. **Ingesting Reference Documents**  
Place your raw .txt reference documents inside the data/documents/ folder, then run the ingestion pipeline to build your vector database:
```text
PowerShell
python -m rag.ingest
```
## How to Run & Test
**Option A: Local CLI Interface**  
Test the end-to-end multi-agent pipeline directly inside your terminal:
```text
PowerShell
python main.py
```
**Option B: FastAPI Web Server (Streaming API)**  
To start the production API server:
```text
PowerShell
uvicorn api.server:app --reload
```
The server will run locally at http://127.0.0.1:8000.

1. **Testing the Live Streaming Event Endpoint (/research)**  
Since PowerShell aliases standard curl commands, execute the POST request via native PowerShell syntax to stream the Server-Sent Events (SSE) logs in real time:
```text
PowerShell
Invoke-RestMethod -Uri "[http://127.0.0.1:8000/research](http://127.0.0.1:8000/research)" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query": "what is langgraph?", "thread_id": "api_test_99"}'
```
2. **Querying State Recovery (/research/{thread_id})**  
To verify that the execution state and checkpoint history was persisted successfully inside data/checkpoints.db:
```text
PowerShell
Invoke-RestMethod -Uri "[http://127.0.0.1:8000/research/api_test_99](http://127.0.0.1:8000/research/api_te
```
