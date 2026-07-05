import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")

def retrieve(query: str, k: int = 4) -> list[str]:
    """Queries local ChromaDB and returns top-k matching string chunks."""
    if not os.path.exists(DB_DIR):
        print("[-] Vector store directory does not exist. Please run ingest.py first.")
        return []

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]