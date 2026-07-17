from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from agent.llm import get_llm
from agent.mcp_client import get_mcp_tools

_graph_cache = None


async def build_graph():
    """Builds (and caches) the compiled agent graph.

    IMPORTANT: this must be awaited from the same event loop that will
    later call graph.ainvoke()/astream() — do not build this at module
    import time with a separate throwaway event loop. The MCP stdio
    connection is bound to whichever loop created it; using it from a
    different loop causes a silent, permanent hang with no error.
    """
    global _graph_cache
    if _graph_cache is not None:
        return _graph_cache

    _, mcp_tools = await get_mcp_tools()

    # Clean wrapper around the MCP retrieve_context tool
    @tool
    async def retrieve_context(query: str, k: int = 5):
        """
        Retrieves relevant information from the knowledge base.
        Use this tool whenever you need to fetch external data to answer a question.
        """
        for t in mcp_tools:
            if t.name == "retrieve_context":
                return await t.ainvoke({"query": query, "k": k})
        return "Error: Context retrieval tool not found."

    safe_tools = [t for t in mcp_tools if t.name != "retrieve_context"]
    safe_tools.append(retrieve_context)

    llm = get_llm(temperature=0.0)

    # NOTE: removed "You must return tool calls in native JSON format." —
    # that instruction was causing the model to write tool calls out as
    # literal text (e.g. "<function=retrieve_context ...>") instead of using
    # Groq's native function-calling, which broke parsing on some questions.
    _graph_cache = create_react_agent(
        model=llm,
        tools=safe_tools,
        prompt=(
            "You are a helpful assistant with two kinds of tools:\n"
            "1. 'retrieve_context' — a knowledge base search tool. Always call this "
            "first when the user asks a question that requires factual or "
            "document-based knowledge, and base your answer on the retrieved "
            "content rather than prior knowledge.\n"
            "2. Filesystem tools (e.g. list_directory, read_file) — these are already "
            "rooted at the project's data directory. To list or read files in that "
            "directory, use '.' (or the exact filename) as the path — do NOT pass "
            "'data' or 'data/' as part of the path, since that directory is already "
            "the tool's root and would double the path and fail. Do not explore "
            "subdirectories (like chroma_db) unless the user specifically asks about "
            "their contents — a single top-level listing is usually sufficient.\n"
            "When a question needs both document knowledge and file listing, call "
            "both tools in the same conversation before giving your final answer."
        )
    )
    return _graph_cache