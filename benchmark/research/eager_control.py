"""Is the EAGER baseline weak because of the architecture or because of the prompt?

    python -m benchmark.research.eager_control --limit 101

The single-step comparison in the manuscript measured 50.0 percent selection
accuracy for EAGER against 96.7 percent for ROUTER, using the prompt
"Execute this task by calling the appropriate tool: {query}" with no system
instruction. That is the weakest number in the study and the obvious objection
is that the baseline was under-specified rather than the architecture inferior.

The ROUTER selection stage in recovery.py receives explicit discrimination
guidance: "Choose the capability whose PURPOSE matches ... Pay attention to the
verb: reading is not writing, searching is not opening one result." Withholding
equivalent guidance from EAGER would make the comparison unfair. The TUNED
condition supplies the same guidance, adapted to the fact that the model is
choosing among 101 declared tools rather than a shortlist.

Conditions, all evaluated on identical queries:

  eager-naive   the original prompt, no system instruction        (reproduction)
  eager-tuned   system instruction with matched guidance
  router-flat   cosine over the shipped index                     (free, cached)
  router-blend  multi-vector retrieval from Section VI            (free, cached)

Sampling is one query per operation, so every capability is represented exactly
once and the result is not dominated by whichever operations happen to have more
queries in the benchmark.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import anyio.to_thread
import numpy as np

import vertex_env
from benchmark.agent_benchmark import get_eager_agent_tools
from benchmark.research.common import Corpus, mcnemar

NAIVE_PROMPT = "Execute this task by calling the appropriate tool: {query}"

# Matched, as closely as the two settings allow, to the guidance the ROUTER
# selection stage receives in benchmark/research/recovery.py.
TUNED_SYSTEM = """\
You select and call the single tool that performs what the user asks.

Many of the available tools are closely related and differ only in purpose. Read
the tool descriptions before choosing, and select the tool whose PURPOSE matches
the request, not merely the tool sharing the most words with it.

Pay attention to the verb. Reading is not writing. Searching for matches is not
opening one known item. Listing what exists is not retrieving history. Extracting
text is not summarising it. Filtering rows is not deleting them.

Call exactly one tool. Do not ask a clarifying question.
"""

TUNED_PROMPT = "{query}"


def call_sync(query: str, tools, system: str | None, template: str,
              retries: int = 4) -> dict:
    from google.genai import types

    last = None
    for attempt in range(retries):
        try:
            started = time.perf_counter()
            config = types.GenerateContentConfig(temperature=0.0, tools=tools)
            if system:
                config.system_instruction = system
            response = vertex_env.client().models.generate_content(
                model=vertex_env.MODEL,
                contents=template.format(query=query),
                config=config,
            )
            calls = response.function_calls or []
            usage = response.usage_metadata
            return {
                "tool": calls[0].name if calls else None,
                "ms": (time.perf_counter() - started) * 1000,
                "prompt_tokens": usage.prompt_token_count if usage else 0,
            }
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    return {"tool": None, "ms": 0.0, "prompt_tokens": 0, "error": str(last)}


async def run_condition(name, queries, tools, system, template, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    out: list[dict | None] = [None] * len(queries)
    done = 0
    started = time.perf_counter()

    async def work(i: int):
        nonlocal done
        async with semaphore:
            out[i] = await anyio.to_thread.run_sync(
                call_sync, queries[i], tools, system, template
            )
        done += 1
        if done % 25 == 0:
            rate = done / (time.perf_counter() - started)
            print(f"    {name}: {done}/{len(queries)} ({rate:.1f}/s)", flush=True)

    await asyncio.gather(*(work(i) for i in range(len(queries))))
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description="Tuned EAGER control")
    parser.add_argument("--limit", type=int, default=101,
                        help="queries to evaluate, sampled one per operation")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    corpus = Corpus()

    # One query per operation, first occurrence, for balanced coverage.
    picked, seen = [], set()
    for i, op in enumerate(corpus.y):
        if op not in seen:
            seen.add(int(op))
            picked.append(i)
        if len(picked) >= args.limit:
            break
    queries = [corpus.queries[i] for i in picked]
    expected_op = np.array([corpus.y[i] for i in picked])
    expected_tool = [corpus.tools[o].tool_id for o in expected_op]

    print(f"{len(queries)} queries, one per operation | {corpus.n_ops} operations\n")

    # --- retrieval conditions: free, from cached vectors -------------------
    flat = corpus.Q[picked] @ corpus.V.T
    router_flat = flat.argmax(axis=1) == expected_op

    from benchmark.research.multivector import build_example_matrix, pool

    E, owner = build_example_matrix(corpus, "positive_examples", "RETRIEVAL_DOCUMENT", "pos_doc")
    N, nowner = build_example_matrix(corpus, "negative_examples", "RETRIEVAL_DOCUMENT", "neg_doc")
    blend = (0.60 * flat
             + 0.40 * pool(corpus.Q[picked] @ E.T, owner, corpus.n_ops, "max")
             - 0.10 * pool(corpus.Q[picked] @ N.T, nowner, corpus.n_ops, "max"))
    router_blend = blend.argmax(axis=1) == expected_op

    # --- eager conditions --------------------------------------------------
    tools = get_eager_agent_tools()
    print("  running eager-naive ...", flush=True)
    naive = await run_condition("naive", queries, tools, None, NAIVE_PROMPT, args.concurrency)
    print("  running eager-tuned ...", flush=True)
    tuned = await run_condition("tuned", queries, tools, TUNED_SYSTEM, TUNED_PROMPT,
                                args.concurrency)

    naive_ok = np.array([(naive[i] or {}).get("tool") == expected_tool[i]
                         for i in range(len(queries))])
    tuned_ok = np.array([(tuned[i] or {}).get("tool") == expected_tool[i]
                         for i in range(len(queries))])
    no_call_naive = sum(1 for r in naive if not (r or {}).get("tool"))
    no_call_tuned = sum(1 for r in tuned if not (r or {}).get("tool"))

    print("\n" + "=" * 66)
    print(f"{'condition':<22}{'accuracy':>10}{'no call':>10}{'tokens':>10}{'ms':>10}")
    print("-" * 66)
    for label, ok, res, nocall in (
        ("eager-naive", naive_ok, naive, no_call_naive),
        ("eager-tuned", tuned_ok, tuned, no_call_tuned),
    ):
        tok = np.mean([(r or {}).get("prompt_tokens", 0) for r in res])
        ms = np.mean([(r or {}).get("ms", 0.0) for r in res])
        print(f"{label:<22}{ok.mean()*100:>9.2f}%{nocall:>10}{tok:>10.0f}{ms:>10.0f}")
    print(f"{'router-flat':<22}{router_flat.mean()*100:>9.2f}%{0:>10}{'n/a':>10}{'n/a':>10}")
    print(f"{'router-blend':<22}{router_blend.mean()*100:>9.2f}%{0:>10}{'n/a':>10}{'n/a':>10}")
    print("=" * 66)

    a, b, p = mcnemar(tuned_ok, naive_ok)
    print(f"\ntuned vs naive:        +{a} / -{b}, p={p:.4f}"
          f"  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    a, b, p = mcnemar(router_flat, tuned_ok)
    print(f"router-flat vs tuned:  +{a} / -{b}, p={p:.4f}"
          f"  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    a, b, p = mcnemar(router_blend, tuned_ok)
    print(f"router-blend vs tuned: +{a} / -{b}, p={p:.4f}"
          f"  {'SIGNIFICANT' if p < 0.05 else 'not significant'}")

    print("\nqueries the tuned prompt fixed relative to naive:")
    shown = 0
    for i in range(len(queries)):
        if tuned_ok[i] and not naive_ok[i]:
            print(f"  {str((naive[i] or {}).get('tool')):<26} -> {expected_tool[i]:<26} "
                  f"{queries[i][:38]}")
            shown += 1
            if shown == 10:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
