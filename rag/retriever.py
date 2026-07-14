import os
import threading
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings

# Clear client caches across system executions
chromadb.api.client.SharedSystemClient.clear_system_cache()

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")

# Thread lock to prevent parallel researcher nodes from creating collections simultaneously
_chroma_lock = threading.Lock()

def retrieve(query: str, k: int = 3) -> list:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    chroma_settings = Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        anonymized_telemetry=False
    )

    # Protect vector store initialization from parallel race conditions
    with _chroma_lock:
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
            client_settings=chroma_settings
        )
    
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]