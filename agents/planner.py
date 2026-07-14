import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import PlanExecuteState, SubTask

class Plan(BaseModel):
    sub_tasks: list[dict] = Field(description="List of 2 to 4 sub-tasks, each with 'id' (e.g. sub_0) and 'question'")
    reasoning: str = Field(description="Reasoning behind this decomposition")

def planner_node(state: PlanExecuteState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    structured_llm = llm.with_structured_output(Plan)

    iteration = state.get("iteration", 0)
    query = state["original_query"]

    if iteration == 0:
        prompt = f"Decompose the following complex query into 2 to 4 distinct, focused sub-questions for research:\n\nQuery: {query}"
    else:
        feedback = state["evaluation"]["feedback"]
        prompt = (
            f"Your previous attempt was rejected. Decompose the original query with specific adjustments based on this feedback.\n"
            f"Feedback: {feedback}\n"
            f"Original Query: {query}"
        )

    result = structured_llm.invoke(prompt)
    
    # Cast Pydantic sub-tasks to typed dictionaries matching state
    formatted_tasks = []
    for item in result.sub_tasks:
        formatted_tasks.append({
            "id": item.get("id", f"sub_{len(formatted_tasks)}"),
            "question": item.get("question", "")
        })

    return {
        "sub_tasks": formatted_tasks,
        "retrieved_docs": [],       # Reset accumulator for clean execution
        "aggregated_context": [],
        "draft": ""
    }