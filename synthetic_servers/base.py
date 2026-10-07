"""Build a real MCP server whose tools are driven by ``data/tools.json``.

Each downstream package is a three-line entrypoint that calls :func:`serve`.
The tool list, names, descriptions, and input schemas all come from the
registry, so the registry stays the single source of truth for both the router
and the servers it routes to.

Tool functions are generated with real typed signatures so the MCP SDK derives
a genuine JSON Schema for each tool -- these servers advertise and validate
arguments exactly like any other MCP server. Only the *execution* is fake.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from orchestrator.registry import ToolRecord, load_registry
from synthetic_servers.synthetic import synthetic_result

TYPE_ANNOTATIONS = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list[Any]",
    "object": "dict[str, Any]",
}


def _make_tool_function(record: ToolRecord):
    """Generate a typed function so the SDK can derive the tool's JSON Schema."""
    required = [(k, v) for k, v in record.input_schema.items() if v.get("required")]
    optional = [(k, v) for k, v in record.input_schema.items() if not v.get("required")]

    parameters = []
    for key, spec in required:
        parameters.append(f"{key}: {TYPE_ANNOTATIONS.get(spec.get('type'), 'str')}")
    for key, spec in optional:
        annotation = TYPE_ANNOTATIONS.get(spec.get("type"), "str")
        parameters.append(f"{key}: {annotation} | None = None")

    argument_items = ", ".join(f"{key!r}: {key}" for key in record.input_schema)
    source = (
        f"def {record.tool_id}({', '.join(parameters)}) -> dict[str, Any]:\n"
        f"    return _execute({{{argument_items}}})\n"
    )

    def _execute(arguments: dict[str, Any]) -> dict[str, Any]:
        return synthetic_result(
            server_id=record.server_id,
            tool_id=record.tool_id,
            arguments=arguments,
            operation_id=record.operation_id,
        )

    namespace: dict[str, Any] = {"Any": Any, "_execute": _execute}
    exec(source, namespace)  # noqa: S102 -- source is built from the trusted registry
    function = namespace[record.tool_id]
    function.__doc__ = record.description
    return function


def _transport_security() -> TransportSecuritySettings:
    """DNS-rebinding protection settings.

    Compose puts every server on one private bridge network reachable only by
    service name, and only the orchestrator is published. Set MCP_ALLOWED_HOSTS
    to a comma-separated list to turn strict Host validation back on.
    """
    allowed = os.getenv("MCP_ALLOWED_HOSTS", "").strip()
    if allowed:
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[h.strip() for h in allowed.split(",") if h.strip()],
        )
    return TransportSecuritySettings(enable_dns_rebinding_protection=False)


def build_server(server_id: str) -> MCPServer:
    """Construct the MCP server for ``server_id`` from the registry."""
    registry = load_registry()
    tools = registry.tools_for_server(server_id)
    if not tools:
        raise SystemExit(
            f"No tools registered for server {server_id!r} in {registry.path}. "
            f"Known servers: {', '.join(registry.server_ids)}"
        )

    server_entry = next(s for s in registry.servers if s["server_id"] == server_id)
    mcp = MCPServer(
        name=server_id,
        title=server_entry.get("name", server_id),
        instructions=(
            f"{server_entry.get('description', '')}\n\n"
            "SYNTHETIC SERVER: every tool returns deterministic fake data and "
            "performs no real external action."
        ),
        version="0.1.0",
    )

    for record in tools:
        mcp.add_tool(
            _make_tool_function(record),
            name=record.tool_id,
            title=record.name,
            description=record.description,
        )

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request):  # noqa: ANN001, ANN202
        from starlette.responses import JSONResponse

        return JSONResponse(
            {
                "status": "ok",
                "server_id": server_id,
                "synthetic": True,
                "tools": [t.tool_id for t in tools],
            }
        )

    return mcp


def serve(server_id: str, default_port: int) -> None:
    """Run ``server_id`` over streamable HTTP at ``/mcp``."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", str(default_port)))

    mcp = build_server(server_id)
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        transport_security=_transport_security(),
    )
    uvicorn.run(app, host=host, port=port, log_level=os.getenv("LOG_LEVEL", "info"))
