import os
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from state import PlanExecuteState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.aggregator import aggregator_node
from agents.writer import writer_node
from agents.evaluator import evaluator_node

# 1. Define Conditional Send Edge (Fan-out)
def dispatch_researchers(state: PlanExecuteState) -> list[Send]:
    """Conditional edge: fan-out one researcher per sub-task."""
    return [
        Send("researcher_node", {**state, "current_sub_task": task})
        for task in state["sub_tasks"]
    ]

# 2. Define Self-Reflection Routing Edge
def route_after_evaluation(state: PlanExecuteState) -> str:
    """Decide whether to stop, route to re-plan, or finish."""
    evaluation = state.get("evaluation", {})
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if iteration >= max_iter:
        return "accept_final"
    
    if evaluation.get("faithfulness", 5) < 3 or evaluation.get("relevance", 5) < 3:
        return "planner_node"  # Loop back to planner with evaluation feedback
    
    return "accept_final"

# 3. Transition helper node to save final answer
def accept_final_node(state: PlanExecuteState) -> dict:
    return {"final_answer": state["draft"]}

# 4. Build StateGraph Flow Topology
builder = StateGraph(PlanExecuteState)

# Add Nodes
builder.add_node("planner_node", planner_node)
builder.add_node("researcher_node", researcher_node)
builder.add_node("aggregator_node", aggregator_node)
builder.add_node("writer_node", writer_node)
builder.add_node("evaluator_node", evaluator_node)
builder.add_node("accept_final_node", accept_final_node)

# Add Edges
builder.add_edge(START, "planner_node")

# Fan-out transition
builder.add_conditional_edges("planner_node", dispatch_researchers)

# Collect / Fan-in transition
builder.add_edge("researcher_node", "aggregator_node")
builder.add_edge("aggregator_node", "writer_node")
builder.add_edge("writer_node", "evaluator_node")

# Routing Loop Transition
builder.add_conditional_edges("evaluator_node", route_after_evaluation, {
    "planner_node": "planner_node",
    "accept_final": "accept_final_node"
})

builder.add_edge("accept_final_node", END)

# 5. Define DB Path variable for importing elsewhere
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "checkpoints.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)