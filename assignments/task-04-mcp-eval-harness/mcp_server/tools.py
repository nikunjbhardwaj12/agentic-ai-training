import os
# Must be set BEFORE importing sentence-transformers/huggingface_hub, and
# BEFORE this module is imported by the MCP server — any progress bar or
# warning text that leaks onto stdout corrupts the MCP JSON-RPC stdio
# protocol, causing tool calls to hang forever with no error.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore")

import datetime
import numexpr as ne
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize the vector store from Task 02/03 persistence path
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)
except Exception:
    vector_store = None

def retrieve(query: str, k: int = 4) -> list[str]:
    """Retrieve top-k relevant text chunks from the local vector database."""
    if vector_store is None:
        return ["Vector store unavailable or uninitialized."]
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

def calculate(expression: str) -> str:
    """Safely evaluates an algebraic math expression using numexpr."""
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters detected in mathematical expression."
        res = ne.evaluate(expression).item()
        return str(res)
    except Exception as e:
        return f"Calculation Error: {str(e)}"

def get_current_date() -> str:
    """Returns today's date formatted as YYYY-MM-DD."""
    return datetime.date.today().strftime("%Y-%m-%d")