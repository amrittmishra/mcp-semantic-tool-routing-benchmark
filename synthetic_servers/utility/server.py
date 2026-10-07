"""Synthetic MCP server: utility_mcp.

Tools, names, and input schemas come from ``data/tools.json``. Every call
returns deterministic fake data -- no real side effect is ever performed.

    python -m synthetic_servers.utility.server
"""

from synthetic_servers.base import serve

SERVER_ID = "utility_mcp"
DEFAULT_PORT = 8118

if __name__ == "__main__":
    serve(SERVER_ID, DEFAULT_PORT)
