import os
import sys
import logging
from mcp.server.fastmcp import FastMCP

# Ensure clean imports from the parent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force logs to stderr so they don't break stdout communication
logging.basicConfig(level=logging.INFO, stream=sys.stderr)

# IMPORTANT: import eagerly, at module load time — BEFORE mcp.run(transport="stdio")
# makes stdout a live JSON-RPC channel. This module's import triggers loading the
# HuggingFace embedding model and connecting to Chroma; if that happened lazily on
# the first tool call instead (as before), any warning/progress-bar text that
# leaked onto stdout mid-call would corrupt the protocol framing and cause the
# call to hang forever with no error.
from mcp_server.tools import retrieve, calculate, get_current_date

# Instantiate FastMCP
mcp = FastMCP("task04-tools")

@mcp.tool()
def retrieve_context(query: str, k: int = 4) -> list:
    """Find document snippets matching a query string.

    Args:
        query: The text search string.
        k: Number of fragments to extract.
    """
    return retrieve(str(query), int(k))

@mcp.tool()
def calculate_expression(expression: str) -> str:
    """Solve safe mathematical equations.

    Args:
        expression: The exact expression string to solve.
    """
    return calculate(str(expression))

@mcp.tool()
def get_today_date() -> str:
    """Get the current calendar date."""
    return get_current_date()

if __name__ == "__main__":
    mcp.run(transport="stdio")