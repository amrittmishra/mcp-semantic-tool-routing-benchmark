"""MCP client for the synthetic downstream servers.

This module is the reason the demo is honest: the orchestrator does **not**
import any downstream tool implementation. It resolves ``server_id`` to an HTTP
endpoint from environment configuration and speaks MCP over streamable HTTP:

    orchestrator MCP server  ->  MCP  ->  downstream synthetic MCP server

Each downstream server gets one long-lived worker task that owns its MCP
session, so the ``initialize`` handshake is paid once per server rather than
once per routed request. Requests are handed to the worker over a queue, which
keeps every session confined to the single task that created it -- anyio cancel
scopes are task-bound, so sharing a session across request tasks is not safe.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

from mcp import Client

from orchestrator import config

# Local (non-Docker) fallback ports, assigned in registry order. Docker Compose
# overrides all of these with service DNS names via *_MCP_URL env vars.
LOCAL_PORTS: dict[str, int] = {
    "filesystem_mcp": 8101,
    "document_mcp": 8102,
    "database_mcp": 8103,
    "web_mcp": 8104,
    "research_mcp": 8105,
    "email_mcp": 8106,
    "calendar_mcp": 8107,
    "messaging_mcp": 8108,
    "git_mcp": 8109,
    "issue_tracker_mcp": 8110,
    "text_mcp": 8111,
    "math_mcp": 8112,
    "data_analysis_mcp": 8113,
    "json_mcp": 8114,
    "system_mcp": 8115,
    "geo_mcp": 8116,
    "media_mcp": 8117,
    "utility_mcp": 8118,
}


class DownstreamError(RuntimeError):
    """Downstream MCP call failed -- distinct from a routing failure."""


def env_var_for(server_id: str) -> str:
    """``filesystem_mcp`` -> ``FILESYSTEM_MCP_URL``."""
    return f"{server_id.upper()}_URL"


def endpoint_for(server_id: str) -> str:
    """Resolve a downstream MCP endpoint URL for ``server_id``."""
    url = os.getenv(env_var_for(server_id))
    if url:
        return url
    port = LOCAL_PORTS.get(server_id)
    if port is None:
        raise DownstreamError(
            f"No endpoint configured for server {server_id!r}. "
            f"Set {env_var_for(server_id)}."
        )
    return f"http://127.0.0.1:{port}/mcp"


def normalize_result(result: Any) -> dict[str, Any]:
    """Turn an MCP ``CallToolResult`` into a plain JSON-able dict."""
    if getattr(result, "structured_content", None):
        payload = result.structured_content
        # MCP wraps non-object returns under "result"; unwrap for readability.
        if isinstance(payload, dict) and set(payload) == {"result"}:
            return payload["result"]
        return payload

    texts = [
        block.text
        for block in getattr(result, "content", []) or []
        if getattr(block, "type", None) == "text"
    ]
    joined = "\n".join(texts)
    if not joined:
        return {}
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return {"text": joined}


class _ServerWorker:
    """Owns one downstream MCP session for the lifetime of the process."""

    def __init__(self, server_id: str, url: str) -> None:
        self.server_id = server_id
        self.url = url
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], asyncio.Future]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ready: asyncio.Future | None = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._task = asyncio.create_task(self._run(), name=f"mcp-{self.server_id}")
        await self._ready

    @property
    def alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        assert self._ready is not None
        try:
            async with Client(self.url, raise_exceptions=True) as client:
                if not self._ready.done():
                    self._ready.set_result(True)
                while True:
                    tool_name, arguments, future = await self._queue.get()
                    if future.cancelled():
                        continue
                    try:
                        result = await client.call_tool(
                            tool_name,
                            arguments,
                            read_timeout_seconds=config.DOWNSTREAM_TIMEOUT_SECONDS,
                        )
                        future.set_result(result)
                    except Exception as exc:  # noqa: BLE001
                        future.set_exception(
                            DownstreamError(
                                f"{self.server_id}.{tool_name} failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- connection setup/teardown
            error = DownstreamError(
                f"MCP connection to {self.server_id} at {self.url} failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if not self._ready.done():
                self._ready.set_exception(error)
            # Fail anything still queued so callers do not hang.
            while not self._queue.empty():
                _, _, future = self._queue.get_nowait()
                if not future.done():
                    future.set_exception(error)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._queue.put((tool_name, arguments, future))
        return await future

    async def aclose(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task


class DownstreamRegistry:
    """server_id -> live MCP session, created lazily and reused."""

    def __init__(self) -> None:
        self._workers: dict[str, _ServerWorker] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, server_id: str) -> asyncio.Lock:
        return self._locks.setdefault(server_id, asyncio.Lock())

    async def _worker(self, server_id: str) -> _ServerWorker:
        async with self._lock(server_id):
            worker = self._workers.get(server_id)
            if worker is not None and worker.alive:
                return worker
            # Either first use, or the previous session died -- reconnect.
            if worker is not None:
                await worker.aclose()
            worker = _ServerWorker(server_id, endpoint_for(server_id))
            await worker.start()
            self._workers[server_id] = worker
            return worker

    async def call_tool(
        self,
        server_id: str,
        tool_id: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        worker = await self._worker(server_id)
        try:
            result = await worker.call(tool_id, arguments)
        except DownstreamError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DownstreamError(
                f"{server_id}.{tool_id} failed: {type(exc).__name__}: {exc}"
            ) from exc
        return normalize_result(result)

    async def list_tools(self, server_id: str) -> list[str]:
        """Diagnostics only -- the orchestrator never exposes these upstream."""
        async with Client(endpoint_for(server_id), raise_exceptions=True) as client:
            result = await client.list_tools()
            return [tool.name for tool in result.tools]

    async def aclose(self) -> None:
        for worker in self._workers.values():
            await worker.aclose()
        self._workers.clear()
