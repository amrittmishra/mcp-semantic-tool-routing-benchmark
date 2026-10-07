"""Full prompt-token cost of ROUTER and EAGER on the three Table 7 requests.

    python -m benchmark.research.table7_chain_cost --spawn-servers --repeats 3

Both arms are Google ADK agents driven by the same model, the same instruction,
and the same 18 synthetic MCP servers; only the tools they are given differ.

  EAGER   one ADK agent with all 101 operations as FunctionTools. Every turn
          carries the 101 schemas plus the accumulated history.
  ROUTER  one ADK agent with a single FunctionTool, ``route_and_execute``.
          When the agent calls it, the orchestrator embeds the request once,
          takes the FAISS top 10, makes ONE planning call over that shortlist
          (which also fills the step arguments), and executes the plan.

The ROUTER total counts every generation prompt on its path: each turn of the
primary agent (the turn that calls the orchestrator and the turn that reads its
result) plus the planning call. The embedding request is not a generation
prompt and QoS selection is arithmetic, so neither adds prompt tokens. Prompt
counts are read from Vertex usage metadata on every call.

Each request is run ``--repeats`` times per arm, in separate fresh ADK
sessions, and every call is written to the JSON log.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid
from typing import Any

import anyio.to_thread

import vertex_env
from orchestrator.downstream import LOCAL_PORTS, DownstreamRegistry
from orchestrator.registry import load_registry
from orchestrator.router import Router

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUTPUT = pathlib.Path(__file__).resolve().parents[1] / "table7_chain_cost.json"

# The chain samples of the evaluation UI, i.e. the three requests in Table 7.
REQUESTS = [
    ("PDF extract, summarize", "Pull the text out of report.pdf and give me a short summary"),
    ("Paper search, cite, summarize",
     "Search for the paper on tool routing, get its citations, and summarize it"),
    ("Fetch, extract, translate",
     "Fetch https://example.com/pricing, extract the text, and translate it to Hindi"),
]

# Shared by both agents, verbatim from the EAGER baseline.
INSTRUCTION = (
    "Complete the user's task by calling the available tools. Call one tool at "
    "a time and use the result of each call to decide the next one. When the "
    "task is complete, reply with a short plain-text summary and do not call "
    "any further tool."
)

ROUTE_TOOL_DESCRIPTION = (
    "Route a natural-language request to the most appropriate available "
    "capability and execute it through the internal MCP capability network. "
    "Describe what you want done in plain language, including any concrete "
    "values (paths, names, dates, numbers) the operation will need."
)

PLAN_SYSTEM = """\
You decompose a request into an ordered plan of tool calls.

You are given the user's request and a shortlist of candidate operations, each
with its id, description, and input schema. Return JSON:

  {"steps": [{"operation_id": "...", "arguments": {...}, "why": "..."}]}

Rules:
- Use ONLY operation_ids from the shortlist.
- Order the steps so each one's prerequisites come first.
- Fill arguments from the request. For a value that only exists after an
  earlier step runs, use the placeholder "$stepN" where N is that step's
  1-based index.
- Omit arguments you cannot fill. Never invent paths, ids, or addresses.
- Use the fewest steps that actually satisfy the request.
"""

SYNTHETIC_MODULES = {
    "filesystem_mcp": "filesystem", "document_mcp": "document", "database_mcp": "database",
    "web_mcp": "web", "research_mcp": "research", "email_mcp": "email",
    "calendar_mcp": "calendar", "messaging_mcp": "messaging", "git_mcp": "git",
    "issue_tracker_mcp": "issues", "text_mcp": "text", "math_mcp": "math",
    "data_analysis_mcp": "data_analysis", "json_mcp": "json_tools",
    "system_mcp": "system", "geo_mcp": "geo", "media_mcp": "media", "utility_mcp": "utility",
}


# --- ROUTER internals (the orchestrator behind the one tool) ----------------

class Orchestrator:
    def __init__(self, downstream: DownstreamRegistry) -> None:
        self.registry = load_registry()
        self.router = Router(registry=self.registry)
        self.client = vertex_env.client()
        self.downstream = downstream
        self.calls: list[dict[str, Any]] = []   # one entry per route_and_execute

    def _plan(self, request_text: str, candidates: list[dict]) -> dict:
        from google.genai import types

        shortlist = []
        for c in candidates:
            tool = self.registry.by_tool_id(c["tool_id"])
            shortlist.append({"operation_id": c["operation_id"],
                              "description": tool.description,
                              "input_schema": tool.json_schema()})
        prompt = (f"User request:\n{request_text}\n\n"
                  f"Candidate operations (retrieved by one embedding):\n"
                  f"{json.dumps(shortlist, indent=2)}\n\nReturn the plan now.")
        response = self.client.models.generate_content(
            model=vertex_env.MODEL, contents=prompt,
            config=types.GenerateContentConfig(system_instruction=PLAN_SYSTEM,
                                               temperature=0.0,
                                               response_mime_type="application/json"))
        usage = response.usage_metadata
        try:
            steps = json.loads(response.text or "{}").get("steps", [])
        except ValueError:
            steps = []
        return {"steps": steps,
                "plan_prompt_tokens": usage.prompt_token_count if usage else None,
                "plan_output_tokens": usage.candidates_token_count if usage else None}

    @staticmethod
    def _resolve(value: Any, outputs: list[Any]) -> Any:
        if isinstance(value, str) and value.startswith("$step"):
            try:
                idx = int(value[5:]) - 1
            except ValueError:
                return value
            if 0 <= idx < len(outputs):
                produced = outputs[idx]
                if isinstance(produced, dict):
                    inner = produced.get("result", produced)
                    if isinstance(inner, dict):
                        for key in ("content", "text", "summary", "value"):
                            if key in inner:
                                return inner[key]
                    return str(inner)[:500]
                return str(produced)[:500]
        return value

    async def route_and_execute(self, request: str) -> dict:
        route = await anyio.to_thread.run_sync(lambda: self.router.route(request, top_k=10))
        candidates = route.candidates
        planned = await anyio.to_thread.run_sync(self._plan, request, candidates)
        by_op = {c["operation_id"]: c for c in candidates}
        steps, outputs = [], []
        for i, step in enumerate(planned["steps"], start=1):
            target = by_op.get(step.get("operation_id"))
            if target is None:
                steps.append({"step": i, "operation_id": step.get("operation_id"),
                              "status": "not_in_shortlist"})
                continue
            arguments = {k: self._resolve(v, outputs)
                         for k, v in (step.get("arguments") or {}).items()}
            try:
                result = await self.downstream.call_tool(target["server_id"],
                                                         target["tool_id"], arguments)
                status = "success"
            except Exception as exc:  # noqa: BLE001
                result, status = {"error": str(exc)}, "failed"
            outputs.append(result)
            steps.append({"step": i, "operation_id": target["operation_id"],
                          "tool_id": target["tool_id"], "status": status, "result": result})
        self.calls.append({"request": request,
                           "shortlist": [c["operation_id"] for c in candidates],
                           "plan_prompt_tokens": planned["plan_prompt_tokens"],
                           "plan_output_tokens": planned["plan_output_tokens"],
                           "planned_operations": [s.get("operation_id") for s in planned["steps"]],
                           "steps": steps})
        # What the primary agent receives back: the executed steps and results.
        return {"steps": [{k: v for k, v in s.items() if k != "tool_id"} for s in steps]}


# --- the two agents ----------------------------------------------------------

async def run_adk(agent, request_text: str, app_name: str) -> dict:
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent, app_name=app_name)
    session_id = str(uuid.uuid4())
    await runner.session_service.create_session(app_name=app_name, user_id="bench",
                                                session_id=session_id)
    turns, final_text, tool_calls = [], None, []
    async for event in runner.run_async(
            user_id="bench", session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=request_text)])):
        usage = getattr(event, "usage_metadata", None)
        if usage:
            turns.append({"prompt_tokens": usage.prompt_token_count or 0,
                          "output_tokens": usage.candidates_token_count or 0})
            if len(turns) > 8:
                break
        content = getattr(event, "content", None)
        for part in (content.parts if content and content.parts else []):
            if getattr(part, "function_call", None):
                tool_calls.append(part.function_call.name)
            if getattr(part, "text", None) and getattr(event, "author", "") != "user":
                final_text = part.text.strip()[:400]
    return {"turns": turns, "tool_calls": tool_calls, "final_text": final_text}


def router_agent(orchestrator: Orchestrator):
    from google.adk.agents import LlmAgent
    from google.adk.tools import FunctionTool

    async def route_and_execute(request: str) -> dict:
        return await orchestrator.route_and_execute(request)

    route_and_execute.__doc__ = ROUTE_TOOL_DESCRIPTION
    return LlmAgent(name="router_agent", model=vertex_env.MODEL, instruction=INSTRUCTION,
                    tools=[FunctionTool(route_and_execute)])


def eager_agent(downstream: DownstreamRegistry):
    try:  # public benchmark package layout
        from benchmark.eager_agent import EagerAgent
    except ImportError:  # full system repository layout
        from ui.eager_agent import EagerAgent
    return EagerAgent(downstream)  # 101 FunctionTools, same INSTRUCTION


# --- synthetic servers -------------------------------------------------------

def use_local_endpoints() -> None:
    """Point every downstream client at the locally spawned servers.

    A repository ``.env`` may define ``<SERVER>_MCP_URL`` with Docker service
    hostnames; left in place, every tool call would fail to connect.
    """
    for server_id in SYNTHETIC_MODULES:
        os.environ[f"{server_id.upper()}_URL"] = (
            f"http://127.0.0.1:{LOCAL_PORTS[server_id]}/mcp")


def spawn_servers() -> list[subprocess.Popen]:
    use_local_endpoints()
    env = dict(os.environ, HOST="127.0.0.1", LOG_LEVEL="warning",
               PYTHONPATH=str(ROOT))
    procs = []
    for server_id, module in SYNTHETIC_MODULES.items():
        procs.append(subprocess.Popen(
            [sys.executable, "-m", f"synthetic_servers.{module}.server"],
            env=dict(env, PORT=str(LOCAL_PORTS[server_id])), cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    import urllib.request
    deadline = time.time() + 60
    for server_id in SYNTHETIC_MODULES:
        url = f"http://127.0.0.1:{LOCAL_PORTS[server_id]}/health"
        while True:
            try:
                if urllib.request.urlopen(url, timeout=2).status == 200:
                    break
            except Exception:  # noqa: BLE001
                if time.time() > deadline:
                    raise SystemExit(f"synthetic server {server_id} did not start")
                time.sleep(0.5)
    return procs


# --- validity --------------------------------------------------------------------

async def preflight(downstream: DownstreamRegistry) -> None:
    """Fail fast unless one real tool call succeeds on every server."""
    registry = load_registry()
    for server_id in SYNTHETIC_MODULES:
        tool = next(t for t in registry.tools if t.server_id == server_id)
        args = {k: "x" for k, v in tool.input_schema.items() if v.get("required")}
        try:
            await downstream.call_tool(server_id, tool.tool_id, args)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"preflight: {server_id} unreachable ({exc}); "
                             "start the servers or pass --spawn-servers") from exc
    print(f"preflight: all {len(SYNTHETIC_MODULES)} MCP servers answered a tool call",
          flush=True)


class InvalidRun(RuntimeError):
    """A completed agent loop whose tool calls did not all execute."""


def router_valid(calls: list[dict]) -> None:
    steps = [s for c in calls for s in c["steps"]]
    if not calls or not steps:
        raise InvalidRun("ROUTER executed no step")
    bad = [s for s in steps if s.get("status") != "success"]
    if bad:
        raise InvalidRun(f"ROUTER: {len(bad)}/{len(steps)} steps failed: {bad[0]}")


def eager_valid(result: dict) -> None:
    if result.get("error"):
        raise InvalidRun(f"EAGER error: {result['error']}")
    executed = result.get("executed") or []
    if not executed:
        raise InvalidRun("EAGER executed no tool")
    bad = [x for x in executed if x.get("status") != "success"]
    if bad:
        raise InvalidRun(f"EAGER: {len(bad)}/{len(executed)} calls failed: {bad[0]}")
    if result.get("truncated"):
        raise InvalidRun("EAGER hit the turn cap")


# --- retries -------------------------------------------------------------------

def _throttled(exc_or_text: Any) -> bool:
    text = str(exc_or_text)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "ResourceExhausted" in text


async def with_retries(make_attempt, check, attempts: int = 12,
                       max_invalid: int = 3) -> dict:
    """Run one arm until it yields a complete, valid agent loop.

    Throttled attempts are discarded and retried. A loop that completes but
    has any failed downstream call is rejected by ``check`` and retried at most
    ``max_invalid`` times; if every attempt is invalid the experiment stops
    rather than record it. ``attempts_used`` and ``rejected`` are logged.
    """
    rejected: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            result = await make_attempt()
            if isinstance(result, dict) and _throttled(result.get("error", "")):
                raise RuntimeError(result["error"])
            check(result)
            result["attempts_used"] = attempt
            result["rejected"] = rejected
            return result
        except InvalidRun as exc:
            rejected.append(str(exc)[:300])
            print(f"    rejected invalid run: {str(exc)[:160]}", flush=True)
            if len(rejected) > max_invalid:
                raise SystemExit(f"{len(rejected)} invalid runs in a row; not recording")
        except Exception as exc:  # noqa: BLE001
            if not _throttled(exc) or attempt == attempts:
                raise
            wait = min(120, 15 * attempt)
            print(f"    throttled (attempt {attempt}), retrying in {wait}s", flush=True)
            await asyncio.sleep(wait)
    raise SystemExit("exhausted attempts")


# --- main ----------------------------------------------------------------------

async def main_async(args) -> dict:
    downstream = DownstreamRegistry()
    await preflight(downstream)
    orchestrator = Orchestrator(downstream)
    eager = eager_agent(downstream)
    rows = []
    for label, text in REQUESTS:
        for rep in range(1, args.repeats + 1):
            async def router_attempt():
                orchestrator.calls.clear()
                return await run_adk(router_agent(orchestrator), text, "router")

            r = await with_retries(
                router_attempt, lambda _: router_valid(orchestrator.calls))
            agent_tokens = sum(t["prompt_tokens"] for t in r["turns"])
            plan_tokens = sum(c["plan_prompt_tokens"] or 0 for c in orchestrator.calls)
            router = {"agent_turns": r["turns"], "agent_prompt_tokens": agent_tokens,
                      "orchestrator_calls": list(orchestrator.calls),
                      "plan_prompt_tokens": plan_tokens,
                      "total_prompt_tokens": agent_tokens + plan_tokens,
                      "final_text": r["final_text"], "attempts_used": r["attempts_used"],
                      "rejected": r["rejected"]}
            e = await with_retries(lambda: eager.run(text), eager_valid)
            row = {"request": label, "text": text, "repeat": rep, "router": router,
                   "eager": {k: e.get(k) for k in ("turns", "turn_count", "tool_calls",
                                                   "operations_called", "executed",
                                                   "prompt_tokens", "final_text",
                                                   "truncated", "error", "attempts_used",
                                                   "rejected")}}
            rows.append(row)
            ratio = (e.get("prompt_tokens") or 0) / max(1, router["total_prompt_tokens"])
            print(f"{label:32s} rep {rep}: ROUTER agent {agent_tokens:5d} "
                  f"({'+'.join(str(t['prompt_tokens']) for t in r['turns'])}) + plan {plan_tokens:5d}"
                  f" = {router['total_prompt_tokens']:5d} | EAGER {e.get('prompt_tokens')} "
                  f"({'+'.join(str(t['prompt_tokens']) for t in e.get('turns') or [])}) "
                  f"| ratio {ratio:.1f}x", flush=True)
            await asyncio.sleep(args.pause)
    return {"model": vertex_env.MODEL, "instruction": INSTRUCTION,
            "validity": "every recorded run executed all of its downstream tool "
                        "calls successfully; runs with any failed call were rejected",
            "requests": REQUESTS, "repeats": args.repeats, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pause", type=float, default=2.0)
    parser.add_argument("--spawn-servers", action="store_true",
                        help="start the 18 synthetic MCP servers on their local ports")
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    args = parser.parse_args()
    procs = spawn_servers() if args.spawn_servers else []
    try:
        out = asyncio.run(main_async(args))
    finally:
        for p in procs:
            p.terminate()
    args.output.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
