import os
import chromadb
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings

# Prevent database lockups on Windows
chromadb.api.client.SharedSystemClient.clear_system_cache()

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")

def ingest_docs():
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
        print(f"Created documents directory at {DOCS_DIR}. Please add .txt files.")
        return

    loader = DirectoryLoader(DOCS_DIR, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    if not documents:
        print("No documents found to index.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Pure-Python Segment API fallback for Windows compatibility
    chroma_settings = Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        anonymized_telemetry=False
    )

    # Store locally
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=DB_DIR,
        client_settings=chroma_settings
    )
    print(f"Successfully ingested {len(splits)} chunks into {DB_DIR}.")

if __name__ == "__main__":
    ingest_docs()