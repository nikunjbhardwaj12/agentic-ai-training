from state import PipelineState
from rag.retriever import retrieve

def researcher_node(state: PipelineState) -> dict:
    """Retrieves context chunks from our localized Vector Database."""
    query = state["query"]
    print(f"\n[Researcher] Querying vector store for: '{query}'")
    
    chunks = retrieve(query, k=4)
    return {"retrieved_docs": chunks, "approved": False}