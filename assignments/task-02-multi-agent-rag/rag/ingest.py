import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = os.path.join(os.path.dirname(__file__), "../data/documents")
DB_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")

def ingest_documents():
    """Reads local documents, chunks them, and embeds them into ChromaDB."""
    if not os.path.exists(DOCS_DIR):
        print(f"[-] Documents directory {DOCS_DIR} not found.")
        return

    # 1. Load Documents
    documents = []
    for file in os.listdir(DOCS_DIR):
        if file.endswith((".txt", ".md")):
            with open(os.path.join(DOCS_DIR, file), "r", encoding="utf-8") as f:
                documents.append(f.read())

    if not documents:
        print("[-] No documents found to index.")
        return

    # 2. Chunk Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.create_documents(documents)

    # 3. Embed and Persist (Idempotent approach: recreating/clearing past instances)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"[+] Ingesting {len(chunks)} text chunks into ChromaDB at {DB_DIR}...")
    Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print("[+] Ingestion complete.")

if __name__ == "__main__":
    ingest_documents()