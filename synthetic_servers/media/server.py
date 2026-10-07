"""Synthetic MCP server: media_mcp.

Tools, names, and input schemas come from ``data/tools.json``. Every call
returns deterministic fake data -- no real side effect is ever performed.

    python -m synthetic_servers.media.server
"""

from synthetic_servers.base import serve

SERVER_ID = "media_mcp"
DEFAULT_PORT = 8117

if __name__ == "__main__":
    serve(SERVER_ID, DEFAULT_PORT)
