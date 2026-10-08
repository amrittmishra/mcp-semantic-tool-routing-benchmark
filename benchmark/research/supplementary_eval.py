"""Re-run the routing comparisons on the expanded 707-request benchmark.

    python -m benchmark.research.supplementary_eval --expanded path/to/expanded_dataset.jsonl \
        [--selector] [--jev]

The expanded file holds the original 505 hand-authored queries unchanged plus
202 model-authored, scenario-based requests (two per operation, ids *_006 and
*_007). This script embeds only the 202 new requests (the 505 query vectors
are reused from the cache), then scores every method on three sets: the
original 505 (which must reproduce the paper's tables exactly), the 202 new
requests alone, and all 707.

Methods and how each treats the new requests:
  flat dense, hierarchy gates, soft prior     no parameters
  BM25, RRF, score interpolation              fixed settings from the paper
  multi-vector                                weights alpha=.60, lambda=.10 chosen
                                              by 5-fold CV on the 505, so the
                                              202 are a fully held-out test; also
                                              7-fold CV over all 707
  generation selector over flat top 5         Vertex AI (--selector)
  Jev over all 101 descriptions               OpenRouter (--jev); the 505 reuse
                                              the saved pass

Every per-request prediction is written to the JSON output.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import pathlib
import time

import anyio.to_thread
import numpy as np

from benchmark.hierarchy_experiment import build_tree, route_gated, route_soft
from benchmark.research.common import Corpus, mcnemar
from benchmark.research.lexical_hybrid import BM25
from benchmark.research.multivector import build_example_matrix, pool
from benchmark.research.tune_blend import BLEND_GRID, NEG_GRID, combined, stratified_folds
from orchestrator import config, embeddings
from orchestrator import index as index_module
from orchestrator.registry import embedding_document

BENCHMARK_DIR = pathlib.Path(__file__).resolve().parents[1]
OUTPUT = pathlib.Path(__file__).resolve().parents[1] / "supplementary_eval.json"
JEV_SAVED = BENCHMARK_DIR / "jev_results.jsonl"


# --- data --------------------------------------------------------------------

def embed_new(queries: list[str]) -> np.ndarray:
    digest = hashlib.sha256("\n".join(queries).encode()).hexdigest()[:16]
    path = config.CACHE_DIR / f"supplementary_query_vectors_{digest}.npy"
    if path.exists():
        return np.load(path)
    vectors = []
    for i, text in enumerate(queries):           # one text per call: the quota is tight
        for attempt in range(40):
            try:
                vectors.append(embeddings.embed_texts([text], task_type="RETRIEVAL_QUERY")[0])
                break
            except Exception as exc:  # noqa: BLE001
                wait = min(60, 3 * (attempt + 1))
                print(f"  embed {i}: {str(exc)[-50:]} -> retry in {wait}s", flush=True)
                time.sleep(wait)
        else:
            raise SystemExit(f"could not embed request {i}")
        time.sleep(0.5)
    Q = index_module.l2_normalize(np.asarray(vectors, dtype=np.float32))
    np.save(path, Q)
    print(f"cached {len(queries)} new query vectors -> {path.name}")
    return Q


# --- statistics --------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(100 * (centre - half), 1), round(100 * (centre + half), 1)


def summarize(correct: np.ndarray, flat: np.ndarray, mask: np.ndarray) -> dict:
    c, f = correct[mask], flat[mask]
    fixed, broke, p = mcnemar(c, f)
    return {"n": int(mask.sum()), "correct": int(c.sum()),
            "top1": round(100 * float(c.mean()), 2), "ci95": wilson(int(c.sum()), int(mask.sum())),
            "fixed": fixed, "broke": broke, "p_vs_flat": p}


# --- retrieval-side methods ---------------------------------------------------

def lexical_methods(corpus: Corpus, queries: list[str], dense: np.ndarray) -> dict[str, np.ndarray]:
    """BM25, RRF over top-5 lists, and min-max interpolation, as in lexical_hybrid."""
    bm25 = BM25([embedding_document(t) for t in corpus.tools])
    L = np.vstack([bm25.scores(q) for q in queries])
    n, m = dense.shape
    bm_order = np.argsort(-L, axis=1, kind="stable")
    d_order = np.argsort(-dense, axis=1)
    out = {"BM25": bm_order[:, 0]}

    def minmax(v):
        span = v.max() - v.min()
        return (v - v.min()) / span if span > 1e-12 else np.zeros_like(v)

    rrf_top, interp = {}, {0.5: [], 0.7: []}
    for qi in range(n):
        d5, b5 = d_order[qi, :5], bm_order[qi, :5]
        dense_rank = {int(j): r + 1 for r, j in enumerate(d5)}
        bm_rank = {int(j): r + 1 for r, j in enumerate(b5)}
        cand = sorted(set(dense_rank) | set(bm_rank))
        rrf = {j: (1 / (60 + dense_rank[j]) if j in dense_rank else 0)
               + (1 / (60 + bm_rank[j]) if j in bm_rank else 0) for j in cand}
        # tie-break by dense score, exactly as the published baseline
        rrf_top[qi] = max(cand, key=lambda j: (rrf[j], dense[qi, j] if j in dense_rank else -1))
        # interpolation over the dense top-5 (missing dense score = 0 after normalisation)
        known = np.array(sorted(dense_rank))
        dn = np.zeros(m)
        dn[known] = minmax(dense[qi, known])
        ln = minmax(L[qi])
        for a in interp:
            s = a * dn + (1 - a) * ln
            tie = np.where(np.isin(np.arange(m), known), dense[qi], -1)
            interp[a].append(int(np.lexsort((-tie, -s))[0]))
    out["RRF, BM25 + dense top 5"] = np.array([rrf_top[i] for i in range(n)])
    out["Interpolated, alpha_h=.5"] = np.array(interp[0.5])
    out["Interpolated, alpha_h=.7"] = np.array(interp[0.7])
    return out


# --- model-side methods -------------------------------------------------------

async def run_selector(corpus: Corpus, queries: list[str], order: np.ndarray,
                       concurrency: int = 4) -> list[dict]:
    from benchmark.research.recovery import select_sync

    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = [None] * len(queries)  # type: ignore[list-item]

    async def work(i: int) -> None:
        cands = [{"idx": int(j), "operation_id": corpus.operations[int(j)]} for j in order[i, :5]]
        async with sem:
            for attempt in range(8):
                try:
                    results[i] = await anyio.to_thread.run_sync(
                        select_sync, queries[i], cands, corpus)
                    return
                except Exception as exc:  # noqa: BLE001
                    await asyncio.sleep(min(60, 5 * (attempt + 1)))
                    last = exc
            results[i] = {"chosen": None, "error": str(last)[:200]}

    await asyncio.gather(*(work(i) for i in range(len(queries))))
    return results


async def run_jev(queries: list[str], ops: list[str], concurrency: int = 4) -> list[dict]:
    import httpx

    from orchestrator.jev import choose, criteria_for
    from orchestrator.registry import load_registry

    criteria = criteria_for(load_registry())
    sem = asyncio.Semaphore(concurrency)
    out: list[dict] = [None] * len(queries)  # type: ignore[list-item]
    async with httpx.AsyncClient(timeout=60) as client:
        async def work(i: int) -> None:
            async with sem:
                try:
                    r = await choose(queries[i], criteria, client=client)
                    out[i] = {"jev_operation": r["operation_id"], "jev_ms": r["latency_ms"],
                              "jev_input_tokens": r["usage"]["input_tokens"],
                              "jev_cost_usd": r["usage"]["cost"]}
                except Exception as exc:  # noqa: BLE001
                    out[i] = {"jev_operation": None, "error": str(exc)[:200]}
        await asyncio.gather(*(work(i) for i in range(len(queries))))
    return out


# --- main ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded", type=pathlib.Path, required=True)
    ap.add_argument("--selector", action="store_true")
    ap.add_argument("--jev", action="store_true")
    ap.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    args = ap.parse_args()

    corpus = Corpus()
    base_ids = [r["id"] for r in corpus.dataset]
    expanded = [json.loads(l) for l in args.expanded.read_text().splitlines() if l.strip()]
    by_id = {r["id"]: r for r in expanded}
    assert all(by_id[r["id"]] == r for r in corpus.dataset), "original 505 must be unchanged"
    new_rows = [r for r in expanded if r["id"] not in set(base_ids)]
    print(f"original {len(corpus.dataset)} | new {len(new_rows)} | total {len(expanded)}")

    Q_new = embed_new([r["query"] for r in new_rows])
    rows = list(corpus.dataset) + new_rows
    queries = [r["query"] for r in rows]
    Q = np.vstack([corpus.Q, Q_new])
    y = np.array([corpus.op_index[r["expected_operation"]] for r in rows])
    is_new = np.array([False] * len(corpus.dataset) + [True] * len(new_rows))
    sets = {"original_505": ~is_new, "new_202": is_new, "all_707": np.ones(len(rows), bool)}

    V = corpus.V
    flat = Q @ V.T
    flat_top1 = flat.argmax(1)
    flat_ok = flat_top1 == y
    preds: dict[str, np.ndarray] = {"Flat dense": flat_top1}
    top3 = {"Flat dense": float(np.mean([y[i] in np.argsort(-flat[i])[:3] for i in range(len(y))]))}

    taxonomy = {t.operation_id: (t.taxonomy_class, t.taxonomy_type) for t in corpus.tools}
    tree = build_tree(V, corpus.operations, taxonomy)
    unreachable = {}
    for level, beam, label in (("class", 1, "Class gate, beam 1"), ("class", 2, "Class gate, beam 2"),
                               ("node", 1, "Node gate, beam 1"), ("node", 3, "Node gate, beam 3")):
        ranked, _, chosen = route_gated(Q, V, tree[level], beam=beam)
        preds[label] = ranked[:, 0]
        members = [set(np.concatenate([tree[level]["members"][n] for n in chosen[q]]).tolist())
                   for q in range(len(Q))]
        unreachable[label] = {k: int(sum(y[i] not in members[i] for i in np.where(m)[0]))
                              for k, m in sets.items()}
    ranked, _, _ = route_soft(Q, V, tree["class"], 0.15)
    preds["Soft class prior, w=.15"] = ranked[:, 0]

    preds.update(lexical_methods(corpus, queries, flat))

    E, owner = build_example_matrix(corpus, "positive_examples", "RETRIEVAL_DOCUMENT", "pos_doc")
    N, nowner = build_example_matrix(corpus, "negative_examples", "RETRIEVAL_DOCUMENT", "neg_doc")
    ex_max = pool(Q @ E.T, owner, corpus.n_ops, "max")
    neg_max = pool(Q @ N.T, nowner, corpus.n_ops, "max")
    mv_fixed = combined(flat, ex_max, neg_max, 0.60, 0.10)
    preds["Multi-vector, weights from 505 CV"] = mv_fixed.argmax(1)
    # 5-fold CV over the original 505 exactly as published, for the reproduction check
    folds505 = stratified_folds(y[~is_new], 5)
    cv505 = np.zeros((~is_new).sum(), dtype=int)
    idx505 = np.where(~is_new)[0]
    for f in range(5):
        tr, te = idx505[folds505 != f], idx505[folds505 == f]
        best = max(((a, l) for a in BLEND_GRID for l in NEG_GRID),
                   key=lambda al: ((combined(flat[tr], ex_max[tr], neg_max[tr], *al).argmax(1) == y[tr]).mean()))
        cv505[folds505 == f] = combined(flat[te], ex_max[te], neg_max[te], *best).argmax(1)
    # 7-fold CV over all 707 (one request per operation per fold)
    folds = stratified_folds(y, 7)
    cv = np.zeros(len(y), dtype=int)
    picks = []
    for f in range(7):
        tr, te = folds != f, folds == f
        best = max(((a, l) for a in BLEND_GRID for l in NEG_GRID),
                   key=lambda al: ((combined(flat[tr], ex_max[tr], neg_max[tr], *al).argmax(1) == y[tr]).mean()))
        picks.append([float(best[0]), float(best[1])])
        cv[te] = combined(flat[te], ex_max[te], neg_max[te], *best).argmax(1)
    preds["Multi-vector, 7-fold CV on 707"] = cv

    out_rows = [{"id": r["id"], "query": r["query"], "expected_operation": r["expected_operation"],
                 "new": bool(is_new[i]),
                 "flat_top5": [corpus.operations[int(j)] for j in np.argsort(-flat[i])[:5]]}
                for i, r in enumerate(rows)]

    if args.selector:
        order = np.argsort(-flat, axis=1)
        sel = asyncio.run(run_selector(corpus, queries, order))
        preds["Selector over flat top 5"] = np.array(
            [corpus.op_index.get((s or {}).get("chosen"), -1) for s in sel])
        for i, s in enumerate(sel):
            out_rows[i]["selector"] = s

    if args.jev:
        saved = {json.loads(l)["id"]: json.loads(l) for l in JEV_SAVED.read_text().splitlines() if l.strip()}
        jev_new = asyncio.run(run_jev([r["query"] for r in new_rows], corpus.operations))
        jev_all = [saved[r["id"]] for r in corpus.dataset] + jev_new
        preds["Jev, all 101 descriptions"] = np.array(
            [corpus.op_index.get(j.get("jev_operation"), -1) for j in jev_all])
        for i, j in enumerate(jev_all):
            out_rows[i]["jev"] = {k: j.get(k) for k in ("jev_operation", "jev_ms", "jev_input_tokens", "jev_cost_usd", "error")}

    results = {}
    for name, p in preds.items():
        ok = p == y
        results[name] = {k: summarize(ok, flat_ok, m) for k, m in sets.items()}
        for i in range(len(rows)):
            out_rows[i].setdefault("predictions", {})[name] = corpus.operations[int(p[i])] if p[i] >= 0 else None
    cv505_ok = cv505 == y[~is_new]
    results["Multi-vector, 5-fold CV on 505 (published)"] = {
        "original_505": summarize(cv505_ok, flat_ok[~is_new], np.ones(len(cv505_ok), bool))}

    # scaling on 707: random subsets as in benchmark.research.scaling
    rng = np.random.default_rng(0)
    scaling = {}
    for size in (10, 20, 30, 50, 70, 90, 101):
        accs = []
        for _ in range(40 if size < 101 else 1):
            sub = np.sort(rng.choice(corpus.n_ops, size, replace=False)) if size < 101 else np.arange(101)
            mask = np.isin(y, sub)
            loc = {g: k for k, g in enumerate(sub)}
            s = Q[mask] @ V[sub].T
            accs.append(float((s.argmax(1) == np.array([loc[g] for g in y[mask]])).mean()))
        scaling[size] = [round(100 * float(np.mean(accs)), 2), round(100 * float(np.std(accs)), 2)]
    xs = np.log(np.array(list(scaling)))
    slope = float(np.polyfit(xs, [v[0] for v in scaling.values()], 1)[0])

    print(f"\n{'method':40s} {'505':>8s} {'202 new':>8s} {'fixed/broke':>11s} {'p':>7s} {'707':>8s}  CI(707)")
    for name, r in results.items():
        o, nw, al = r.get("original_505"), r.get("new_202"), r.get("all_707")
        if nw:
            print(f"{name:40s} {o['top1']:7.2f}% {nw['top1']:7.2f}% {nw['fixed']:>5d}/{nw['broke']:<5d} {nw['p_vs_flat']:7.3f} {al['top1']:7.2f}%  {al['ci95']}")
        else:
            print(f"{name:40s} {o['top1']:7.2f}%")
    print(f"flat top-3: 505 {np.mean([y[i] in np.argsort(-flat[i])[:3] for i in np.where(~is_new)[0]])*100:.2f}%"
          f" | 202 {np.mean([y[i] in np.argsort(-flat[i])[:3] for i in np.where(is_new)[0]])*100:.2f}%"
          f" | 707 {top3['Flat dense']*100:.2f}%")
    print("unreachable:", unreachable)
    print("7-fold picks:", picks)
    print(f"scaling on 707: {scaling}  slope {slope:.2f} points per e-fold")

    args.output.write_text(json.dumps({
        "expanded_file": str(args.expanded), "n": {k: int(m.sum()) for k, m in sets.items()},
        "results": results, "unreachable": unreachable, "multivector_7fold_picks": picks,
        "flat_top3": {k: round(100 * float(np.mean([y[i] in np.argsort(-flat[i])[:3] for i in np.where(m)[0]])), 2)
                      for k, m in sets.items()},
        "scaling_707": {"by_size": scaling, "slope_per_efold": round(slope, 2)},
        "rows": out_rows}, indent=1))
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
