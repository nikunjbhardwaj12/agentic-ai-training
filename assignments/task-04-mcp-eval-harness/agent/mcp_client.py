import os
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient


def _build_mcp_config():
    current_python = sys.executable
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    server_script = os.path.join(project_root, "mcp_server", "server.py")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)

    env_copy = os.environ.copy()
    env_copy["PYTHONPATH"] = project_root

    task04_tools_config = {
        "command": current_python,
        "args": [server_script],
        "transport": "stdio",
        "env": env_copy,
    }

    if os.name == "nt":
        filesystem_config = {
            "command": "cmd.exe",
            "args": ["/c", "npx", "-y", "@modelcontextprotocol/server-filesystem", data_dir],
            "transport": "stdio",
            "env": env_copy,
        }
    else:
        filesystem_config = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", data_dir],
            "transport": "stdio",
            "env": env_copy,
        }

    return {
        "task04_tools": task04_tools_config,
        "filesystem": filesystem_config,
    }


async def get_mcp_tools():
    """Must be awaited from the SAME event loop that will later invoke the
    returned tools. Do not create this via run_until_complete() on a
    throwaway loop and then use the tools from a different loop later —
    the stdio transport is bound to the loop it was created on, and using
    it from another loop causes a silent hang (no exception, no timeout)."""
    client = MultiServerMCPClient(_build_mcp_config())
    tools = await client.get_tools()
    return client, tools