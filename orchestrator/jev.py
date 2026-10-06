"""Jev operation selection through OpenRouter's Decisions API.

Jev chooses a semantic operation. The registry remains responsible for mapping
that operation to a concrete tool and MCP server.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from orchestrator.registry import Registry

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"


def criteria_for(registry: Registry) -> dict[str, str]:
    return {
        tool.operation_id: tool.description
        for tool in registry.tools
    }


DEFAULT_INSTRUCTIONS = (
    "Which single operation best fulfills the user's request? "
    "Choose by the requested action and object, not by a shared keyword."
)


def request_body(query: str, criteria: dict[str, str],
                 instructions: str = DEFAULT_INSTRUCTIONS) -> dict[str, Any]:
    return {
        "model": MODEL,
        "state": {"request": query},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }
        },
    }


def api_key() -> str:
    # vertex_env loads the repo's .env; OP is the OpenRouter key for Jev.
    import vertex_env  # noqa: F401

    key = os.getenv("OP")
    if not key:
        raise RuntimeError("Set OP in .env for OpenRouter Jev routing")
    return key


async def choose(
    query: str,
    criteria: dict[str, str],
    *,
    client: httpx.AsyncClient | None = None,
    instructions: str = DEFAULT_INSTRUCTIONS,
) -> dict[str, Any]:
    started = time.perf_counter()
    own_client = client is None
    client = client or httpx.AsyncClient(timeout=45)
    try:
        for attempt in range(5):
            response = await client.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {api_key()}"},
                json=request_body(query, criteria, instructions),
            )
            if response.status_code not in {429, 500, 502, 503, 504, 524, 529} or attempt == 4:
                response.raise_for_status()
                break
            await asyncio.sleep(0.5 * 2 ** attempt)
        payload = response.json()
        answer = payload["answers"]["operation"]
        choice = answer["choice"]
        if choice not in criteria:
            raise ValueError(f"Jev returned unknown operation {choice!r}")
        probabilities = answer.get("probabilities", {})
        usage = payload.get("usage", {})
        return {
            "operation_id": choice,
            "confidence": answer.get("confidence", probabilities.get(choice)),
            "probabilities": probabilities,
            "model": payload.get("model"),
            "usage": {
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cost": float(usage["cost"]) if usage.get("cost") is not None else None,
            },
            "latency_ms": (time.perf_counter() - started) * 1000,
        }
    finally:
        if own_client:
            await client.aclose()
