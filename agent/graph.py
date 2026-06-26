from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import agent_node, summarizer_node
from agent.tools import all_tools

# Initialize the StateGraph
workflow = StateGraph(AgentState)

# Add our custom processing nodes
workflow.add_node("agent", agent_node)
workflow.add_node("summarizer", summarizer_node)

# Use LangGraph's prebuilt ToolNode to manage tool execution automatically
tool_node = ToolNode(all_tools)
workflow.add_node("tools", tool_node)

# Set up operational routing
workflow.add_edge(START, "agent")

# Use prebuilt tools_condition to check if the agent requested a tool execution
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "tools",  # If LLM wants a tool, go to 'tools' node
        "__end__": "summarizer"  # If LLM wants to talk/finish, route to markdown cleanup
    }
)

# After tools run, we always circle back to the agent to interpret the result data
workflow.add_edge("tools", "agent")
workflow.add_edge("summarizer", END)

# Complete compilation with a lightweight memory buffer checkpointer for multi-turn chats
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)