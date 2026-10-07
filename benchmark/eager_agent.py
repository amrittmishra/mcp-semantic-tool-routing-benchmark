"""The EAGER baseline, built on Google's Agent Development Kit.

    from benchmark.eager_agent import EagerAgent

An earlier version of this comparison hand-rolled the agent loop, which invites
the obvious objection: "your baseline was badly implemented." This runs the
EAGER arm on `google-adk` -- Google's own agent framework, against Vertex AI --
so the loop, the tool-calling protocol, and the history management are all
Google's, not ours. The only thing we supply is the tool catalogue and the
instrumentation.

All 101 downstream capabilities are declared to the model as ADK FunctionTools.
Each one really executes: it calls the corresponding synthetic MCP server over
MCP and returns its response to the agent, so the loop is a genuine
observe-act cycle rather than a single scripted turn.

Token accounting is taken from the usage metadata on each ADK event, so
"prompt tokens" is what Vertex actually billed across every turn, including the
conversation history that accumulates as the chain proceeds. That is precisely
the cost a one-turn projection cannot see.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import vertex_env
from orchestrator.downstream import DownstreamRegistry
from orchestrator.registry import ToolRecord, load_registry

TYPE_ANNOTATIONS = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

INSTRUCTION = (
    "Complete the user's task by calling the available tools. Call one tool at "
    "a time and use the result of each call to decide the next one. When the "
    "task is complete, reply with a short plain-text summary and do not call "
    "any further tool."
)


def _make_tool_function(record: ToolRecord, downstream: DownstreamRegistry, log: list):
    """Build an async function ADK can expose, wired to the real MCP server."""
    required = [(k, v) for k, v in record.input_schema.items() if v.get("required")]
    optional = [(k, v) for k, v in record.input_schema.items() if not v.get("required")]

    params = [f"{k}: {TYPE_ANNOTATIONS.get(v.get('type'), 'str')}" for k, v in required]
    params += [
        f"{k}: {TYPE_ANNOTATIONS.get(v.get('type'), 'str')} = None" for k, v in optional
    ]
    args_literal = ", ".join(f"{k!r}: {k}" for k in record.input_schema)

    async def _execute(arguments: dict[str, Any]) -> dict[str, Any]:
        arguments = {k: v for k, v in arguments.items() if v is not None}
        started = time.perf_counter()
        try:
            result = await downstream.call_tool(
                record.server_id, record.tool_id, arguments
            )
            status = "success"
        except Exception as exc:  # noqa: BLE001
            result, status = {"error": str(exc)}, "failed"
        log.append({
            "tool": record.tool_id,
            "operation_id": record.operation_id,
            "server_id": record.server_id,
            "arguments": arguments,
            "status": status,
            "ms": (time.perf_counter() - started) * 1000,
            "result": result,
        })
        return result

    source = (
        f"async def {record.tool_id}({', '.join(params)}) -> dict:\n"
        f"    return await _execute({{{args_literal}}})\n"
    )
    namespace: dict[str, Any] = {"_execute": _execute}
    exec(source, namespace)  # noqa: S102 -- built from the trusted registry
    fn = namespace[record.tool_id]
    fn.__doc__ = record.description
    return fn


class EagerAgent:
    """101 tool schemas in context, driven by ADK's own agent loop."""

    def __init__(self, downstream: DownstreamRegistry, max_turns: int = 8) -> None:
        from google.adk.agents import LlmAgent
        from google.adk.runners import InMemoryRunner
        from google.adk.tools import FunctionTool

        self.registry = load_registry()
        self.downstream = downstream
        self.max_turns = max_turns
        self._log: list[dict] = []

        tools = [
            FunctionTool(_make_tool_function(record, downstream, self._log))
            for record in self.registry.tools
        ]
        self.agent = LlmAgent(
            name="eager_agent",
            model=vertex_env.MODEL,
            instruction=INSTRUCTION,
            tools=tools,
        )
        self.runner = InMemoryRunner(self.agent, app_name="eager")
        self.tool_count = len(tools)

    async def run(self, request_text: str) -> dict:
        """Run to completion; report what Vertex actually billed."""
        from google.genai import types

        self._log.clear()
        wall = time.perf_counter()
        user_id = "bench"
        session_id = str(uuid.uuid4())
        await self.runner.session_service.create_session(
            app_name="eager", user_id=user_id, session_id=session_id
        )

        turns: list[dict] = []
        prompt_tokens = output_tokens = 0
        final_text = None
        truncated = False

        try:
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user", parts=[types.Part.from_text(text=request_text)]
                ),
            ):
                usage = getattr(event, "usage_metadata", None)
                if usage:
                    # One LLM turn: the full schema block plus everything the
                    # conversation has accumulated so far.
                    p = usage.prompt_token_count or 0
                    o = usage.candidates_token_count or 0
                    prompt_tokens += p
                    output_tokens += o
                    turns.append({"turn": len(turns) + 1, "prompt_tokens": p,
                                  "output_tokens": o})
                    if len(turns) > self.max_turns:
                        truncated = True
                        break

                content = getattr(event, "content", None)
                if content and getattr(content, "parts", None):
                    for part in content.parts:
                        if getattr(part, "text", None) and getattr(event, "author", "") != "user":
                            final_text = part.text.strip()[:400]
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}", "measured": True}

        executed = list(self._log)
        return {
            "measured": True,
            "framework": "google-adk",
            "turns": turns,
            "turn_count": len(turns),
            "tool_calls": len(executed),
            "tools_called": [c["tool"] for c in executed],
            "operations_called": [c["operation_id"] for c in executed],
            "executed": executed,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "downstream_ms": sum(c["ms"] for c in executed),
            "total_ms": (time.perf_counter() - wall) * 1000,
            "tool_schemas_in_context": self.tool_count,
            "final_text": final_text,
            "truncated": truncated,
        }
