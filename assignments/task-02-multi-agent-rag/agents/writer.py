from langchain_groq import ChatGroq
from state import PipelineState

def writer_node(state: PipelineState) -> dict:
    """Generates the final structured synthesis complete with Markdown source citations."""
    print("\n[Writer] Drafting final structured answer...")
    # Swapped ChatOpenAI to ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    context = "\n\n".join([f"[Source Chunk {i+1}]: {doc}" for i, doc in enumerate(state["retrieved_docs"])])
    
    prompt = f"""
    You are an expert technical writer. Answer the user's query comprehensively using ONLY the verified context chunks provided below. 
    You MUST cite your sources using inline tags like [Source Chunk 1] corresponding to where you drew information from.

    Context Chunks:
    {context}

    User Question:
    {state['query']}
    """
    
    response = llm.invoke(prompt)
    return {"draft": response.content, "final_answer": response.content}