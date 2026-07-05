from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from state import PipelineState

# Import node functions from your agents package
from agents.supervisor import supervisor_node
from agents.researcher import researcher_node
from agents.writer import writer_node

# Initialize the state graph builder
workflow = StateGraph(PipelineState)

# Add all agent nodes to the workflow
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("writer", writer_node)

# Core structural layout edges
workflow.add_edge(START, "supervisor")
workflow.add_edge("researcher", "supervisor")
workflow.add_edge("writer", "supervisor")

# Dynamic conditional routing engine mapping strings to destinations
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "researcher": "researcher",
        "writer": "writer",
        "end": END
    }
)

# Compile with an in-memory checkpointer AND an explicit human review break point
memory = InMemorySaver()
graph = workflow.compile(
    checkpointer=memory,
    interrupt_after=["researcher"]  # Force graph to pause instantly when researcher updates state
)