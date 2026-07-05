import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.tools import all_tools

load_dotenv()

# We use the recommended Llama model from Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Bind the tools to the model
llm_with_tools = llm.bind_tools(all_tools)

def agent_node(state: AgentState) -> dict:
    """Calls the LLM to decide on an action (call tool or talk to user)."""
    messages = state["messages"]
    
    # Prepend a helpful system prompt if it's the beginning of a fresh state context
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_message = SystemMessage(
            content="You are an expert conversational Research Agent. Use your tools to look up "
                    "accurate information. If you've collected tool data and are ready to provide an answer, "
                    "or if you have enough info, just respond directly. Always state what you are doing."
        )
        messages = [sys_message] + messages

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def summarizer_node(state: AgentState) -> dict:
    """Condenses search information into a beautiful structured markdown summary."""
    messages = state["messages"]
    
    summary_prompt = (
        "Review the conversation history and specifically any tool outputs provided above. "
        "Synthesize the findings into a clear, clean, structured summary using beautiful Markdown formatting. "
        "Ensure you address the user's explicit question directly."
    )
    
    # Append a instruction message specifically directed to summarizing
    summary_input = messages + [HumanMessage(content=summary_prompt)]
    summary_response = llm.invoke(summary_input)
    
    return {
        "messages": [summary_response],
        "final_summary": summary_response.content
    }