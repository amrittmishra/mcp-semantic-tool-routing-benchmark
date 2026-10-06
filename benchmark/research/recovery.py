"""Does planning over a shortlist convert top-k recall into top-1 accuracy?

    python -m benchmark.research.recovery --k 3
    python -m benchmark.research.recovery --k 5 --limit 120

The gap this targets: flat routing gets 95.64% top-1 but 99.60% top-3. On 20 of
the 22 queries it gets wrong, the correct operation is sitting in the top 3 --
retrieval already found it, and only the ordering is wrong.

Plan-then-route never commits to the top-1. It hands the LLM the whole shortlist
and lets it choose. So the question is whether the LLM can pick correctly from a
set retrieval has already narrowed to k, and how much of that 4-point gap it
closes.

The decisive numbers are not raw accuracy but the PAIRED ones:
  recovered  flat was wrong, the planner is right   (the upside)
  broken     flat was right, the planner is wrong   (the cost)
A technique that recovers 20 and breaks 20 is worthless however good its
headline accuracy looks.

Query vectors come from cache, so retrieval costs nothing here and only the
selection call hits Vertex. Run this in a process that never imports google-adk:
its instrumentation inflates every Vertex call roughly tenfold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

import anyio.to_thread
import numpy as np

import vertex_env
from benchmark.research.common import Corpus, mcnemar

SYSTEM = """\
You select which single capability a user request needs.

You are given the request and a shortlist of candidate capabilities, each with
an id and a description. Exactly one of them is correct.

Return only: {"operation_id": "<the id you choose>"}

Choose the capability whose PURPOSE matches what the user is asking for, not
merely the one sharing the most words with the request. Pay attention to the
verb: reading is not writing, searching is not opening one result, listing is
not fetching history.
"""


def build_prompt(query: str, candidates: list[dict], corpus: Corpus) -> str:
    lines = [f"User request:\n{query}\n", "Candidate capabilities:"]
    for c in candidates:
        tool = corpus.tools[c["idx"]]
        lines.append(f"- {c['operation_id']}: {tool.description}")
    lines.append("\nWhich one does the request need?")
    return "\n".join(lines)


def select_sync(query: str, candidates: list[dict], corpus: Corpus,
                retries: int = 4) -> dict:
    """One selection call, with backoff.

    Concurrency draws 429s from Vertex, and a failed call silently becomes a
    wrong answer -- which would be scored as the technique failing rather than
    the harness failing. Retrying keeps measurement error out of the result.
    """
    last = None
    for attempt in range(retries):
        try:
            return _select_once(query, candidates, corpus)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"selection failed after {retries} attempts: {last}")


def _select_once(query: str, candidates: list[dict], corpus: Corpus) -> dict:
    from google.genai import types

    started = time.perf_counter()
    response = vertex_env.client().models.generate_content(
        model=vertex_env.MODEL,
        contents=build_prompt(query, candidates, corpus),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    usage = response.usage_metadata
    try:
        chosen = json.loads(response.text or "{}").get("operation_id")
    except ValueError:
        chosen = None
    return {
        "chosen": chosen,
        "ms": (time.perf_counter() - started) * 1000,
        "prompt_tokens": usage.prompt_token_count if usage else 0,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Shortlist-planning recovery experiment")
    parser.add_argument("--k", type=int, default=3, help="shortlist size")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--retrieval", choices=("flat", "blend"), default="flat",
                        help="how the shortlist is built")
    args = parser.parse_args()

    corpus = Corpus()
    flat_scores = corpus.Q @ corpus.V.T

    if args.retrieval == "blend":
        # Multi-vector retrieval: the operation's summary document PLUS one
        # vector per positive example, minus similarity to its negatives.
        # Weights are the ones 5-fold cross-validation selected (alpha=0.60 in
        # 4 of 5 folds, lambda=0.10 in all 5) -- NOT the in-sample optimum.
        from benchmark.research.multivector import build_example_matrix, pool

        E, owner = build_example_matrix(
            corpus, "positive_examples", "RETRIEVAL_DOCUMENT", "pos_doc")
        N, nowner = build_example_matrix(
            corpus, "negative_examples", "RETRIEVAL_DOCUMENT", "neg_doc")
        ex_max = pool(corpus.Q @ E.T, owner, corpus.n_ops, "max")
        neg_max = pool(corpus.Q @ N.T, nowner, corpus.n_ops, "max")
        scores = 0.60 * flat_scores + 0.40 * ex_max - 0.10 * neg_max
    else:
        scores = flat_scores

    order = np.argsort(-scores, axis=1)
    flat_order = np.argsort(-flat_scores, axis=1)

    n = len(corpus.y) if args.limit is None else min(args.limit, len(corpus.y))
    idx = list(range(n))

    # Baseline is always plain flat top-1 -- the shipped system -- so every
    # variant is compared against the same reference point.
    flat_correct = np.array([flat_order[i, 0] == corpus.y[i] for i in idx])
    retrieval_top1 = np.array([order[i, 0] == corpus.y[i] for i in idx])
    in_topk = np.array([corpus.y[i] in order[i, : args.k] for i in idx])

    print(f"{n} queries | {corpus.n_ops} operations | shortlist k={args.k} "
          f"| retrieval={args.retrieval}")
    print(f"flat top-1 (baseline):      {flat_correct.mean()*100:.2f}%")
    print(f"{args.retrieval} top-1 (no planner):  {retrieval_top1.mean()*100:.2f}%")
    print(f"top-{args.k} recall (ceiling for any reranker): {in_topk.mean()*100:.2f}%")
    print(f"headroom: {(in_topk.mean()-flat_correct.mean())*100:.2f} points\n")

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict | None] = [None] * n
    done = 0
    started_all = time.perf_counter()

    async def work(i: int) -> None:
        nonlocal done
        candidates = [
            {"idx": int(j), "operation_id": corpus.operations[int(j)],
             "score": float(scores[i, j])}
            for j in order[i, : args.k]
        ]
        async with semaphore:
            try:
                out = await anyio.to_thread.run_sync(
                    select_sync, corpus.queries[i], candidates, corpus
                )
            except Exception as exc:  # noqa: BLE001
                out = {"chosen": None, "ms": 0.0, "prompt_tokens": 0,
                       "error": f"{type(exc).__name__}: {exc}"}
        results[i] = out
        done += 1
        if done % 50 == 0:
            rate = done / (time.perf_counter() - started_all)
            print(f"  {done}/{n}  ({rate:.1f}/s, ~{(n-done)/rate:.0f}s left)", flush=True)

    await asyncio.gather(*(work(i) for i in idx))

    planner_correct = np.array([
        (results[i] or {}).get("chosen") == corpus.operations[corpus.y[i]] for i in idx
    ])
    off_shortlist = sum(
        1 for i in idx
        if (results[i] or {}).get("chosen") not in
        [corpus.operations[int(j)] for j in order[i, : args.k]]
    )
    errors = sum(1 for i in idx if (results[i] or {}).get("error"))

    recovered, broken, p = mcnemar(planner_correct, flat_correct)
    r2, b2, p2 = mcnemar(planner_correct, retrieval_top1)
    latencies = [(results[i] or {}).get("ms", 0.0) for i in idx]
    tokens = [(results[i] or {}).get("prompt_tokens", 0) for i in idx]

    print("\n" + "=" * 62)
    print(f"flat top-1          {flat_correct.mean()*100:6.2f}%   (shipped baseline)")
    print(f"{args.retrieval} retrieval only {retrieval_top1.mean()*100:6.2f}%   (no planner)")
    print(f"plan over top-{args.k}     {planner_correct.mean()*100:6.2f}%  "
          f"({(planner_correct.mean()-flat_correct.mean())*100:+.2f} points)")
    print(f"ceiling (top-{args.k})     {in_topk.mean()*100:6.2f}%")
    print("-" * 62)
    print(f"recovered  {recovered:>4}   flat wrong -> planner right")
    print(f"broken     {broken:>4}   flat right -> planner wrong")
    print(f"McNemar p = {p:.4f}   {'SIGNIFICANT' if p < 0.05 else 'not significant'}")
    if args.retrieval != "flat":
        print(f"  vs {args.retrieval}-only: +{r2} / -{b2}, p={p2:.4f}  "
              f"(what the PLANNER adds on top of better retrieval)")
    print("-" * 62)

    # Of the queries retrieval COULD have saved, how many did the planner save?
    salvageable = ~flat_correct & in_topk
    saved = (planner_correct & salvageable).sum()
    print(f"salvageable (flat wrong but answer in top-{args.k}): {salvageable.sum()}")
    print(f"  of those, planner got right: {saved}/{salvageable.sum()}"
          f"  ({saved/max(1,salvageable.sum())*100:.0f}%)")

    lost_cause = ~flat_correct & ~in_topk
    print(f"unreachable (answer not in top-{args.k} at all): {lost_cause.sum()}")
    print("-" * 62)
    print(f"cost: {np.mean(tokens):.0f} prompt tokens, {np.mean(latencies):.0f} ms "
          f"per selection call")
    if off_shortlist:
        print(f"planner named something off the shortlist: {off_shortlist}")
    if errors:
        print(f"failed calls: {errors}")

    print("\nQueries the planner recovered:")
    shown = 0
    for i in idx:
        if planner_correct[i] and not flat_correct[i]:
            print(f"  {corpus.operations[order[i,0]]:<24} -> "
                  f"{corpus.operations[corpus.y[i]]:<24} {corpus.queries[i][:44]}")
            shown += 1
            if shown == 12:
                break

    print("\nQueries the planner broke:")
    shown = 0
    for i in idx:
        if flat_correct[i] and not planner_correct[i]:
            print(f"  {corpus.operations[corpus.y[i]]:<24} -> "
                  f"{str((results[i] or {}).get('chosen')):<24} {corpus.queries[i][:44]}")
            shown += 1
            if shown == 12:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
