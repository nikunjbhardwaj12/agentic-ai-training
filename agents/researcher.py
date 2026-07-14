from state import PlanExecuteState
from rag.retriever import retrieve

def researcher_node(state: PlanExecuteState) -> dict:
    # State holds 'current_sub_task' context injected dynamically by the Send() dispatcher
    task = state["current_sub_task"]
    chunks = retrieve(task["question"], k=3)
    return {"retrieved_docs": chunks}