from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import PipelineState

class RouteDecision(BaseModel):
    next: Literal["researcher", "writer", "end"] = Field(  # <-- Changed back to "end"
        description="The next node to route execution to based on state evaluation."
    )
    reasoning: str = Field(description="Explanation for making this routing decision.")

def supervisor_node(state: PipelineState) -> dict:
    """Evaluates pipeline state and dictates the next agent execution step."""
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = f"""
    You are an orchestrator routing a multi-agent RAG workflow.
    
    Current State Analysis:
    - User Query: {state.get('query')}
    - Retrieved Chunks Count: {len(state.get('retrieved_docs', []))}
    - Human Approved Flag: {state.get('approved')}
    - Written Draft: {'Present' if state.get('draft') else 'Empty'}

    Rules:
    1. If 'retrieved_docs' is empty, you MUST route to 'researcher'.
    2. If 'retrieved_docs' are present but 'approved' is False/None, you MUST route to 'end' to let the Human-in-the-Loop validation step fire.
    3. If 'approved' is True and 'draft' is empty, route to 'writer'.
    4. If 'draft' is completed, route to 'end'.
    """
    
    decision = structured_llm.invoke(prompt)
    return {"next_agent": decision.next}