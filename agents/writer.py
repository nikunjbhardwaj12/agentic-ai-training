from langchain_groq import ChatGroq
from state import PlanExecuteState

def writer_node(state: PlanExecuteState) -> dict:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    
    context_str = "\n".join([f"[Chunk {i}]: {doc}" for i, doc in enumerate(state["aggregated_context"])])
    feedback_str = f"Feedback to correct: {state['evaluation']['feedback']}" if state.get("iteration", 0) > 0 else ""

    prompt = (
        f"Synthesize an answer to the User Query using the compiled context below.\n"
        f"IMPORTANT: You must cite sources inline using format [Chunk N].\n\n"
        f"User Query: {state['original_query']}\n"
        f"{feedback_str}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Answer:"
    )

    response = llm.invoke(prompt)
    return {"draft": response.content}