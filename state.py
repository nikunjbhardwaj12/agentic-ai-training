import operator
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages

class SubTask(TypedDict):
    id: str          # e.g., "sub_0", "sub_1"
    question: str    # The focused sub-question

class EvaluationResult(TypedDict):
    faithfulness: int  # 1–5 score
    relevance: int     # 1–5 score
    feedback: str      # Critique to improve planning or writing

class PlanExecuteState(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str                 # Original user prompt
    sub_tasks: List[SubTask]            # Decomposed sub-tasks
    retrieved_docs: Annotated[list, operator.add]  # Dynamically concatenates parallel RAG inputs
    aggregated_context: List[str]       # Deduplicated unique references
    draft: str                          # Drafted answer
    final_answer: str                   # Approved final answer
    evaluation: EvaluationResult        # Latest evaluation feedback
    iteration: int                      # Reflection tracker loop counter
    max_iterations: int                 # Safeguard loop cap