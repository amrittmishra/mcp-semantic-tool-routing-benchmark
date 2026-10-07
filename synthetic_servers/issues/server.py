"""Synthetic MCP server: issue_tracker_mcp.

Tools, names, and input schemas come from ``data/tools.json``. Every call
returns deterministic fake data -- no real side effect is ever performed.

    python -m synthetic_servers.issues.server
"""

from synthetic_servers.base import serve

SERVER_ID = "issue_tracker_mcp"
DEFAULT_PORT = 8110

if __name__ == "__main__":
    serve(SERVER_ID, DEFAULT_PORT)
