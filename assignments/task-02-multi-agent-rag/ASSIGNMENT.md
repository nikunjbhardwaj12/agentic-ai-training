# Task 02: Multi-Agent RAG with Human-in-the-Loop (HITL)

An implementation of an Agentic Retrieval-Augmented Generation (RAG) system using **LangGraph** and **LangChain**. The system features a supervisor-led multi-agent architecture with an explicit human-in-the-loop review gate to approve, reject, or rewrite search queries before generation occurs.

---

## System Architecture

The workflow models complex, non-linear LLM tasks using a cyclical state graph layout managed by a centralized Supervisor:

   [START]  
      │  
      ▼  
┌──────────────┐  
│  Supervisor  │◄────────────────────────┐  
└──────────────┘                         │  
│      │     ▲                           │  
│      │     └───────────┐               │  
▼      ▼                 │               │  
┌────────────┐          ┌──────────┐   ┌──────────┐  
│ Researcher │          │  Human   │   │  Writer  │  
└────────────┘          │  Review  │   └──────────┘  
│                       └──────────┘        │  
▼                             ▲             ▼  
[State Intercept] ────────────┘           [END]  


### Core Components
1. **Supervisor Node**: The orchestrator. Evaluates the global pipeline state and dictates which node executes next using structured LLM choices (`llama-3.3-70b-versatile`).
2. **Researcher Node**: Queries the local ChromaDB vector store using `sentence-transformers/all-MiniLM-L6-v2` embeddings to gather contextual data.
3. **Human-in-the-Loop Intercept**: Freezes graph execution right after the researcher updates the state, preventing automated generation until a human reviews the source citations.
4. **Writer Node**: Pulls approved background context and generates a structured, factual final response with source references.

---

## Setup & Installation

### 1. Prerequisites
Ensure you have Python 3.10+ and a virtual environment configured.

### 2. Install Dependencies 
*bash*  
pip install -U langchain-huggingface langgraph langchain-groq chromadb pydantic python-dotenv
### 3. Environment Configuration
Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_read_token_here

---
## Execution Guide
Run the main interactive script from your workspace terminal:

*Bash*  
python -u main.py  
**Interactive Human Review Options:**  
- Enter y: Approves the raw context chunks. The Supervisor routes control to the Writer to draft the final response.

- Enter r <new query>: Rejects chunks, updates the state with a modified query, and routes back to the Researcher for a fresh lookup.

- Enter Any other key: Safely halts execution, rejecting the pipeline flow.

---
## Verification Run Logs
Sample Successful Execution
Plaintext
[*] Checking for document indexing updating...
[+] Ingestion complete.

Enter your search prompt/question: What is LangGraph?

[Researcher] Querying vector store for: 'What is LangGraph?'

HUMAN REVIEW REQUIRED  
The researcher found 4 raw context chunks:  
[1] LangGraph is a library built on top of LangChain designed for creating stateful, multi-agent applications...  

Approve these references? Enter 'y' (yes), 'n' (no), or 'r <new query>': y  

[Writer] Drafting final structured answer...  

FINAL ANSWER OUTPUT  
LangGraph is a library built on top of LangChain, designed for creating stateful, multi-agent applications [Source Chunk 1]. It models agent workflows as cyclical graphs, which are networks with loops, unlike traditional linear chains [Source Chunk 2]...

---