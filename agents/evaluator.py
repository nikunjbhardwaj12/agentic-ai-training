from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from state import PlanExecuteState

class EvaluationResult(BaseModel):
    faithfulness: int = Field(..., description="Score 1-5: Are claims backed completely by context?", ge=1, le=5)
    relevance: int = Field(..., description="Score 1-5: Does this answer user's exact question?", ge=1, le=5)
    feedback: str = Field(..., description="Actionable critique for planner and writer if score < 3")

def evaluator_node(state: PlanExecuteState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    structured_llm = llm.with_structured_output(EvaluationResult)

    context_str = "\n".join(state["aggregated_context"])
    
    prompt = (
        f"Analyze the draft response below based strictly on the retrieved context.\n"
        f"Grade both criteria from 1 (poor) to 5 (excellent).\n\n"
        f"Context:\n{context_str}\n\n"
        f"Draft Response:\n{state['draft']}\n\n"
        f"Provide evaluation structured output."
    )

    result = structured_llm.invoke(prompt)
    current_iteration = state.get("iteration", 0)

    return {
        "evaluation": result.model_dump(),
        "iteration": current_iteration + 1
    }