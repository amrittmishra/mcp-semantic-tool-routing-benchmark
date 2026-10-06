"""EAGER vs ROUTER: what a large tool catalogue costs the agent's context.

    python -m benchmark.agent_benchmark --queries 40

Two ways to give one agent access to 101 capabilities:

  EAGER   all 101 downstream tool schemas are declared to the model. The model
          picks a tool by reasoning over the whole catalogue.
  ROUTER  exactly one tool is declared -- route_and_execute. The model states
          what it wants in plain language; a FAISS taxonomy index picks the
          capability behind the scenes.

What is measured, precisely:

  * prompt tokens reported by Vertex for each call -- the real context cost.
  * eager_llm_ms:  time for the model to emit a tool call over 101 schemas.
  * routed_llm_ms: time for the model to emit route_and_execute.
  * router_ms:     embedding + FAISS lookup, measured separately.
  * routed_total_ms = routed_llm_ms + router_ms, so the router's own cost is
    never hidden inside the comparison.

Downstream MCP execution is NOT included on either side: both architectures
would pay it identically, so including it would only dilute the contrast. The
generated report says so explicitly rather than implying end-to-end numbers.

Correctness is scored on different-but-comparable targets, and the report says
which is which: EAGER is right when it names the expected tool_id; ROUTER is
right when the index selects the expected operation_id. In Branch 1 these are
one-to-one, so the comparison is fair.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import vertex_env
from orchestrator.registry import load_registry
from orchestrator.router import Router

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parent
DATASET_PATH = BENCHMARK_DIR / "dataset.jsonl"
REPORT_PDF_PATH = BENCHMARK_DIR / "benchmark_report.pdf"
RESULTS_PATH = BENCHMARK_DIR / "agent_results.json"

EAGER_COLOR = "#dc2626"
ROUTED_COLOR = "#059669"

ROUTE_TOOL_DESCRIPTION = (
    "Route a natural-language request to the most appropriate available "
    "capability and execute it through the internal MCP capability network. "
    "Describe what you want done in plain language, including any concrete "
    "values (paths, names, dates, numbers) the operation will need."
)


# --- tool declarations ------------------------------------------------------


def _gemini_schema(tool_record):
    from google.genai import types

    properties = {}
    required = []
    for key, spec in tool_record.input_schema.items():
        declared = spec.get("type", "string").upper()
        field = {"type": declared}
        if "description" in spec:
            field["description"] = spec["description"]
        if declared == "ARRAY":
            field["items"] = types.Schema(type="STRING")
        if declared == "OBJECT":
            # An untyped object; Gemini needs the key present.
            field["properties"] = {}
        properties[key] = types.Schema(**field)
        if spec.get("required"):
            required.append(key)

    if not properties:
        return types.Schema(type="OBJECT", properties={})
    return types.Schema(type="OBJECT", properties=properties, required=required or None)


def get_eager_agent_tools():
    """All 101 downstream schemas, flattened into the model's context."""
    from google.genai import types

    registry = load_registry()
    declarations = [
        types.FunctionDeclaration(
            name=tool.tool_id,
            description=tool.description,
            parameters=_gemini_schema(tool),
        )
        for tool in registry.tools
    ]
    return [types.Tool(function_declarations=declarations)]


def get_routed_agent_tools():
    """The single tool the orchestrator actually advertises over MCP."""
    from google.genai import types

    return [
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="route_and_execute",
                    description=ROUTE_TOOL_DESCRIPTION,
                    parameters=types.Schema(
                        type="OBJECT",
                        properties={
                            "request": types.Schema(
                                type="STRING",
                                description=(
                                    "Plain-language description of what you want done."
                                ),
                            )
                        },
                        required=["request"],
                    ),
                )
            ]
        )
    ]


# --- measurement ------------------------------------------------------------


def _call(client, tools, prompt):
    from google.genai import types

    started = time.perf_counter()
    response = client.models.generate_content(
        model=vertex_env.MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, tools=tools),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    usage = response.usage_metadata
    calls = response.function_calls or []
    return {
        "latency_ms": elapsed_ms,
        "prompt_tokens": usage.prompt_token_count if usage else None,
        "selected": calls[0].name if calls else None,
        "arguments": dict(calls[0].args) if calls else {},
    }


def run_benchmark(num_queries: int, sleep_seconds: float) -> dict:
    client = vertex_env.client()
    eager_tools = get_eager_agent_tools()
    routed_tools = get_routed_agent_tools()

    # One Router for the whole run. Constructing it per query would re-read
    # tools.faiss from disk every time and silently inflate router_ms -- the
    # exact number this benchmark exists to report.
    router = Router()

    dataset = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Every Nth row, so the sample spans all 101 operations instead of just
    # the first few servers in registry order.
    stride = max(1, len(dataset) // num_queries)
    queries = dataset[::stride][:num_queries]

    rows = []
    print(f"EAGER vs ROUTER over {len(queries)} queries "
          f"({router.index.size} operations indexed, model={vertex_env.MODEL})\n")

    for i, item in enumerate(queries, 1):
        query = item["query"]
        print(f"[{i}/{len(queries)}] {query[:58]}")

        eager = _call(client, eager_tools,
                      f"Execute this task by calling the appropriate tool: {query}")

        routed = _call(client, routed_tools,
                       f"Execute this task by calling the available tool: {query}")

        started = time.perf_counter()
        route = router.route(query)
        router_ms = (time.perf_counter() - started) * 1000

        rows.append(
            {
                "id": item["id"],
                "query": query,
                "expected_tool": item["expected_tool"],
                "expected_operation": item["expected_operation"],
                "eager": {
                    **eager,
                    "correct": eager["selected"] == item["expected_tool"],
                },
                "routed": {
                    **routed,
                    "router_ms": router_ms,
                    "embedding_ms": route.embedding_ms,
                    "faiss_ms": route.faiss_ms,
                    "total_ms": routed["latency_ms"] + router_ms,
                    "routed_operation": route.top["operation_id"] if route.top else None,
                    "router_status": route.status,
                    "correct": bool(route.top)
                    and route.top["operation_id"] == item["expected_operation"],
                },
            }
        )

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return {"model": vertex_env.MODEL, "indexed_operations": router.index.size, "rows": rows}


def summarize(run: dict) -> dict:
    rows = run["rows"]
    n = len(rows)

    def mean(path, side):
        values = [r[side][path] for r in rows if r[side].get(path) is not None]
        return statistics.fmean(values) if values else 0.0

    eager_tokens = mean("prompt_tokens", "eager")
    routed_tokens = mean("prompt_tokens", "routed")
    eager_ms = mean("latency_ms", "eager")
    routed_llm_ms = mean("latency_ms", "routed")
    router_ms = mean("router_ms", "routed")

    return {
        "model": run["model"],
        "indexed_operations": run["indexed_operations"],
        "queries": n,
        "eager_prompt_tokens": eager_tokens,
        "routed_prompt_tokens": routed_tokens,
        "token_reduction": (1 - routed_tokens / eager_tokens) if eager_tokens else 0.0,
        "eager_llm_ms": eager_ms,
        "routed_llm_ms": routed_llm_ms,
        "router_ms": router_ms,
        "routed_total_ms": routed_llm_ms + router_ms,
        "eager_accuracy": sum(r["eager"]["correct"] for r in rows) / n,
        "routed_accuracy": sum(r["routed"]["correct"] for r in rows) / n,
        "eager_no_call_rate": sum(1 for r in rows if r["eager"]["selected"] is None) / n,
        "routed_no_call_rate": sum(1 for r in rows if r["routed"]["selected"] is None) / n,
        "mean_embedding_ms": mean("embedding_ms", "routed"),
        "mean_faiss_ms": mean("faiss_ms", "routed"),
    }


# --- report -----------------------------------------------------------------


def _charts(summary: dict, rows: list[dict]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = {}

    # 1. context cost
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(
        [f"EAGER\n({summary['indexed_operations']} tool schemas)", "ROUTER\n(1 tool schema)"],
        [summary["eager_prompt_tokens"], summary["routed_prompt_tokens"]],
        color=[EAGER_COLOR, ROUTED_COLOR],
        width=0.55,
    )
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_ylabel("Mean prompt tokens per request")
    ax.set_title("Context cost of the tool catalogue")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(summary["eager_prompt_tokens"], 1) * 1.18)
    fig.tight_layout()
    paths["tokens"] = str(BENCHMARK_DIR / "tokens.png")
    fig.savefig(paths["tokens"], dpi=150)
    plt.close(fig)

    # 2. latency, with the router's own cost shown rather than hidden
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(["EAGER", "ROUTER"], [summary["eager_llm_ms"], summary["routed_llm_ms"]],
           color=[EAGER_COLOR, ROUTED_COLOR], width=0.55, label="LLM tool-call")
    ax.bar(["EAGER", "ROUTER"], [0, summary["router_ms"]],
           bottom=[0, summary["routed_llm_ms"]], color="#93c5fd", width=0.55,
           label="embedding + FAISS")
    for x, total in enumerate([summary["eager_llm_ms"], summary["routed_total_ms"]]):
        ax.text(x, total * 1.02, f"{total:.0f} ms", ha="center", fontsize=9)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Time to select a capability")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(summary["eager_llm_ms"], summary["routed_total_ms"]) * 1.2)
    fig.tight_layout()
    paths["latency"] = str(BENCHMARK_DIR / "latency.png")
    fig.savefig(paths["latency"], dpi=150)
    plt.close(fig)

    # 3. accuracy
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(
        ["EAGER\n(picks tool_id)", "ROUTER\n(picks operation_id)"],
        [summary["eager_accuracy"] * 100, summary["routed_accuracy"] * 100],
        color=[EAGER_COLOR, ROUTED_COLOR], width=0.55,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Correct selection (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Capability selection accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paths["accuracy"] = str(BENCHMARK_DIR / "accuracy.png")
    fig.savefig(paths["accuracy"], dpi=150)
    plt.close(fig)

    return paths


def _verdict(summary: dict) -> str:
    """Describe what was actually measured -- no pre-written conclusion."""
    parts = []

    reduction = summary["token_reduction"] * 100
    if reduction > 0:
        parts.append(
            f"ROUTER cut mean prompt tokens by {reduction:.1f}% "
            f"({summary['eager_prompt_tokens']:.0f} to "
            f"{summary['routed_prompt_tokens']:.0f})."
        )
    else:
        parts.append(
            f"ROUTER did not reduce prompt tokens in this run "
            f"({summary['routed_prompt_tokens']:.0f} vs "
            f"{summary['eager_prompt_tokens']:.0f})."
        )

    eager_ms, routed_ms = summary["eager_llm_ms"], summary["routed_total_ms"]
    if routed_ms < eager_ms:
        parts.append(
            f"End to end it was faster: {routed_ms:.0f} ms against {eager_ms:.0f} ms, "
            f"and that figure already includes the {summary['router_ms']:.0f} ms "
            "embedding and FAISS lookup."
        )
    else:
        parts.append(
            f"It was NOT faster in this run: {routed_ms:.0f} ms against "
            f"{eager_ms:.0f} ms, because the {summary['router_ms']:.0f} ms "
            "embedding call is a network round trip that EAGER does not pay."
        )

    delta = (summary["routed_accuracy"] - summary["eager_accuracy"]) * 100
    if abs(delta) < 2:
        parts.append(
            f"Selection accuracy was comparable "
            f"({summary['routed_accuracy'] * 100:.1f}% vs "
            f"{summary['eager_accuracy'] * 100:.1f}%)."
        )
    elif delta > 0:
        parts.append(
            f"ROUTER selected correctly more often, by {delta:.1f} points "
            f"({summary['routed_accuracy'] * 100:.1f}% vs "
            f"{summary['eager_accuracy'] * 100:.1f}%)."
        )
    else:
        parts.append(
            f"ROUTER selected correctly LESS often, by {abs(delta):.1f} points "
            f"({summary['routed_accuracy'] * 100:.1f}% vs "
            f"{summary['eager_accuracy'] * 100:.1f}%)."
        )

    return " ".join(parts)


def generate_report(summary: dict, rows: list[dict]) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    charts = _charts(summary, rows)
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    story = []

    story.append(Paragraph("Remote MCP Taxonomy Routing", styles["Title"]))
    story.append(Paragraph("EAGER vs ROUTER: the context cost of a large tool catalogue",
                           styles["Heading3"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("1. What was measured", styles["Heading2"]))
    story.append(Paragraph(
        f"Both arms were given the same {summary['queries']} natural-language requests, "
        f"sampled across all {summary['indexed_operations']} semantic operations in the "
        f"catalogue, and answered by <b>{summary['model']}</b> at temperature 0.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>EAGER</b> declares all downstream tool schemas to the model, which must "
        "reason over the whole catalogue to pick one. <b>ROUTER</b> declares exactly one "
        "tool, route_and_execute; the request is embedded with gemini-embedding-001 and "
        "matched against a persistent FAISS index of capability descriptions.", body))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Scope, stated plainly:</b> downstream MCP execution is excluded from both arms. "
        "Both architectures would pay that cost identically, so including it would dilute "
        "the contrast rather than inform it. The latency figures below are time-to-capability-"
        "selection, not end-to-end task completion. The router's own embedding and FAISS "
        "cost is measured separately and added to the ROUTER total rather than omitted.", body))
    story.append(Spacer(1, 14))

    table = Table(
        [
            ["Metric", "EAGER", "ROUTER"],
            ["Tool schemas in context", str(summary["indexed_operations"]), "1"],
            ["Mean prompt tokens", f"{summary['eager_prompt_tokens']:.0f}",
             f"{summary['routed_prompt_tokens']:.0f}"],
            ["Mean LLM latency", f"{summary['eager_llm_ms']:.0f} ms",
             f"{summary['routed_llm_ms']:.0f} ms"],
            ["Embedding + FAISS", "n/a", f"{summary['router_ms']:.0f} ms"],
            ["Mean total latency", f"{summary['eager_llm_ms']:.0f} ms",
             f"{summary['routed_total_ms']:.0f} ms"],
            ["Selection accuracy", f"{summary['eager_accuracy'] * 100:.1f}%",
             f"{summary['routed_accuracy'] * 100:.1f}%"],
            ["Emitted no tool call", f"{summary['eager_no_call_rate'] * 100:.1f}%",
             f"{summary['routed_no_call_rate'] * 100:.1f}%"],
        ],
        colWidths=[2.5 * inch, 1.7 * inch, 1.7 * inch],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f4f6")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))
    story.append(Paragraph("<b>Result:</b> " + _verdict(summary), body))

    story.append(PageBreak())
    story.append(Paragraph("2. Context cost", styles["Heading2"]))
    story.append(Image(charts["tokens"], width=6.0 * inch, height=3.3 * inch))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Every EAGER request re-sends all {summary['indexed_operations']} tool schemas. "
        f"That is a fixed toll of roughly "
        f"{summary['eager_prompt_tokens'] - summary['routed_prompt_tokens']:.0f} tokens per "
        "call, paid whether the agent uses one capability or none. The ROUTER schema is "
        "constant: adding the 200th capability to the index does not change what the agent "
        "sees.", body))
    story.append(Spacer(1, 14))

    story.append(Paragraph("3. Latency", styles["Heading2"]))
    story.append(Image(charts["latency"], width=6.0 * inch, height=3.3 * inch))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"The ROUTER bar is stacked: {summary['routed_llm_ms']:.0f} ms for the model to emit "
        f"route_and_execute, plus {summary['router_ms']:.0f} ms for retrieval "
        f"({summary['mean_embedding_ms']:.0f} ms of which is the Vertex embedding round trip "
        f"and {summary['mean_faiss_ms']:.2f} ms the FAISS search itself). The FAISS component "
        "is effectively free; the embedding call is the real cost, and it is a network round "
        "trip rather than computation.", body))

    story.append(PageBreak())
    story.append(Paragraph("4. Accuracy", styles["Heading2"]))
    story.append(Image(charts["accuracy"], width=6.0 * inch, height=3.3 * inch))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The two arms are scored on different but equivalent targets: EAGER is correct when "
        "it names the expected tool_id, ROUTER when the index selects the expected "
        "operation_id. In Branch 1 exactly one tool implements each operation, so the two "
        "are one-to-one and the comparison is fair.", body))
    story.append(Spacer(1, 14))

    story.append(Paragraph("5. Per-query detail", styles["Heading2"]))
    detail = [["Request", "Expected", "EAGER", "ROUTER"]]
    for row in rows[:26]:
        detail.append([
            Paragraph(f"<font size=7>{row['query'][:60]}</font>", body),
            Paragraph(f"<font size=7>{row['expected_operation']}</font>", body),
            "correct" if row["eager"]["correct"] else (row["eager"]["selected"] or "no call"),
            "correct" if row["routed"]["correct"] else (row["routed"]["routed_operation"] or "-"),
        ])
    detail_table = Table(detail, colWidths=[2.7 * inch, 1.5 * inch, 1.1 * inch, 1.1 * inch])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(detail_table)

    story.append(Spacer(1, 14))
    story.append(Paragraph("6. What this does and does not show", styles["Heading2"]))
    story.append(Paragraph(
        "It shows that the agent-facing schema stays constant while the internal capability "
        "count grows, and quantifies the token toll EAGER pays for that catalogue. It does "
        "not show end-to-end task success: neither arm executed a downstream tool here, and "
        "the synthetic servers would return fabricated data if they had. Routing quality at "
        f"full scale is measured separately over all 505 benchmark queries "
        "(python -m benchmark.evaluate).", body))

    SimpleDocTemplate(str(REPORT_PDF_PATH), pagesize=letter).build(story)


def main() -> int:
    parser = argparse.ArgumentParser(description="EAGER vs ROUTER agent benchmark")
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Pause between queries to stay under rate limits")
    parser.add_argument("--report-only", action="store_true",
                        help="Rebuild the PDF from the last saved run")
    args = parser.parse_args()

    if args.report_only:
        if not RESULTS_PATH.exists():
            print(f"No saved run at {RESULTS_PATH}", file=sys.stderr)
            return 1
        run = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        run = run_benchmark(args.queries, args.sleep)
        RESULTS_PATH.write_text(json.dumps(run, indent=2), encoding="utf-8")

    summary = summarize(run)
    print("\n" + "=" * 62)
    for key, value in summary.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"{key:<26} {formatted}")
    print("=" * 62)
    print("\n" + _verdict(summary))

    generate_report(summary, run["rows"])
    print(f"\nReport: {REPORT_PDF_PATH}")
    print(f"Raw run: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
