"""Paired Jev versus cached FAISS routing benchmark, without tool execution.

Run: python -m benchmark.jev_comparison --concurrency 8
Both arms use the 505 labelled requests in benchmark/dataset.jsonl. The FAISS
arm reads benchmark/results.jsonl, avoiding new embedding calls. Jev receives
all 101 operation descriptions on each request. Results resume by query ID.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import statistics
from collections import Counter

import httpx

from orchestrator.jev import choose, criteria_for
from orchestrator.registry import load_registry

ROOT = pathlib.Path(__file__).resolve().parent
DATASET = ROOT / "dataset.jsonl"
FAISS = ROOT / "results.jsonl"
OUTPUT = ROOT / "jev_results_openrouter.jsonl"


def read_rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summary(rows: list[dict]) -> dict:
    complete = [row for row in rows if "jev_operation" in row]
    valid = [row for row in complete if row["jev_operation"] is not None]
    return {
        "queries": len(rows),
        "jev_completed": len(complete),
        "jev_valid": len(valid),
        "faiss_top1_correct": sum(r["faiss_operation"] == r["expected_operation"] for r in rows),
        "faiss_accepted_correct": sum(r["faiss_status"] == "ok" and r["faiss_operation"] == r["expected_operation"] for r in rows),
        "faiss_accepted": sum(r["faiss_status"] == "ok" for r in rows),
        "jev_correct": sum(r["jev_operation"] == r["expected_operation"] for r in valid),
        "jev_errors": dict(Counter(r.get("error", "") for r in complete if r["jev_operation"] is None)),
        "jev_mean_ms": round(statistics.mean(r["jev_ms"] for r in valid), 1) if valid else None,
        "faiss_mean_ms": round(statistics.mean(r["faiss_ms"] for r in rows), 1) if rows else None,
        "jev_input_tokens": sum(r.get("jev_input_tokens") or 0 for r in valid),
        "jev_cost_usd": round(sum(r.get("jev_cost_usd") or 0 for r in valid), 6),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="First N queries; 0 means all")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        parser.error("concurrency must be 1..32")

    dataset = read_rows(DATASET)
    if args.limit:
        dataset = dataset[: args.limit]
    faiss = {r["id"]: r for r in read_rows(FAISS)}
    existing = {r["id"]: r for r in read_rows(OUTPUT)} if OUTPUT.exists() else {}
    criteria = criteria_for(load_registry())
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async with httpx.AsyncClient(timeout=60) as client:
        async def run(row: dict) -> dict:
            previous = existing.get(row["id"])
            if previous and previous.get("jev_operation") is not None:
                return previous
            base = faiss[row["id"]]
            result = {
                "id": row["id"], "query": row["query"],
                "expected_operation": row["expected_operation"],
                "faiss_operation": base["predicted_operation"],
                "faiss_status": base["status"],
                "faiss_ms": base["total_ms"],
            }
            async with semaphore:
                for attempt in range(4):
                    try:
                        answer = await choose(row["query"], criteria, client=client)
                        result.update(
                            jev_operation=answer["operation_id"],
                            jev_confidence=answer["confidence"],
                            jev_ms=round(answer["latency_ms"], 1),
                            jev_model=answer["model"],
                            jev_input_tokens=answer["usage"].get("input_tokens"),
                            jev_cost_usd=answer["usage"].get("cost"),
                        )
                        break
                    except (httpx.HTTPError, ValueError, KeyError) as exc:
                        if attempt == 3:
                            result.update(jev_operation=None, error=f"{type(exc).__name__}: {exc}")
                        else:
                            await asyncio.sleep(2 ** attempt)
            async with lock:
                with OUTPUT.open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
            return result

        rows = await asyncio.gather(*(run(row) for row in dataset))
    print(json.dumps(summary(rows), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
