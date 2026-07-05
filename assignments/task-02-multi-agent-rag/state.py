from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages

class PipelineState(TypedDict):
    messages: Annotated[list, add_messages]  # Conversation history
    query: str                               # User's original question
    retrieved_docs: List[str]                # Chunks from vector store
    approved: bool                           # Did user approve the context?
    draft: str                               # Writer's drafted answer
    final_answer: str                        # Polished final output
    next_agent: str                          # Supervisor routing decision