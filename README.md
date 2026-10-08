# MCP Semantic Tool Routing Benchmark

Benchmark data, saved experiment outputs, and evaluation code for

> *Scalable Semantic Routing and QoS-Aware Provider Selection for MCP Agents.*
> A. Mishra, Y. Panditrao, S. Parab, A. Sharma, P. Mishra, I. Saha.
> International Conference on Machine Learning and Data Engineering (ICMLDE), Procedia Computer Science, 2026.

Project page: https://mcp-tool-routing-benchmark.vercel.app

The benchmark asks whether a natural-language request can be mapped to one of
**101 semantic operations** spread over **18 synthetic MCP servers**, using
**505 hand-authored queries** (five per operation). The registry, queries,
gold labels, the cached embedding vectors, and every script behind the paper's
tables are included, so all retrieval results reproduce offline.

## Contents

| Path | What it is |
|---|---|
| `dataset/` | Release v1.0.0 as flat CSV + XLSX: `servers.csv` (18), `tools.csv` (101), `queries.csv` (505), `tool_examples.csv` (606), `tool_parameters.csv` (160), data dictionary, dataset card, validation report, checksums |
| `data/tools.json` | The registry the orchestrator loads (same content as `dataset/`, nested) |
| `data/providers.json` | Synthetic provider profiles used by the QoS simulations |
| `data/cache/` | Rebuilt FAISS index (`tools.faiss`, 101 x 768, `gemini-embedding-001`), the 505 query vectors, 303 + 303 example vectors, and 24 compound-request vectors. These make every retrieval experiment runnable without any API key |
| `benchmark/dataset.jsonl` | The 505 queries with gold operation / tool / server |
| `benchmark/dataset_expanded.jsonl`, `dataset/queries_supplementary.csv` | The 505 plus 202 model-authored, scenario-based requests (seven per operation, 707 total); the CSV holds the 202 alone |
| `benchmark/results.jsonl` | Saved flat-FAISS run: top-5 candidates and scores per query (Table 1 baseline, 483/505) |
| `benchmark/jev_results.jsonl`, `jev_results_openrouter.jsonl` | Two independent Jev 1.13 passes over the 505 queries (494/505 each) |
| `benchmark/qos_dynamic_results.json`, `qos_dynamic_optimistic.json` | Saved dynamic QoS simulation outputs |
| `orchestrator/` | The minimal library the scripts import: registry loader, embedding document builder, FAISS index, router, MCP client, Jev client, QoS registry and simulator |
| `synthetic_servers/` | The 18 synthetic MCP servers (deterministic fabricated outputs, no side effects) used by the Table 8 harness |
| `benchmark/table7_chain_cost.json`, `benchmark/qos_sensitivity_results.json` | Saved Table 8 runs and the full QoS sensitivity grid |
| `benchmark/`, `benchmark/research/` | Experiment scripts (table below) |

## Reproducing the paper's tables

Python 3.11+. Install and run from the repository root:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

Not every result can be reproduced to the same degree. Each row says which.

**Offline: no API key, from the shipped embeddings, index, and simulator**

| Paper result | Command |
|---|---|
| Table 1 flat vs. hierarchical; Table 2 authored vs. k-means gates | `python -m benchmark.hierarchy_experiment` |
| Table 3 accuracy vs. catalogue size | `python -m benchmark.research.scaling` |
| Table 4 mean-centering / ABTT / CSLS rows | `python -m benchmark.research.reranking` |
| Table 4 multi-vector 2-fold and 5-fold rows | `python -m benchmark.research.tune_blend` |
| Multi-vector ablations (Sec. 5.2) | `python -m benchmark.research.multivector` |
| Table 5 BM25, RRF, interpolation rows | `python -m benchmark.research.lexical_hybrid` |
| Table 6 offline rows on the 202 supplementary requests and all 707 (flat, gates, soft prior, BM25, RRF, interpolation, multi-vector) | `python -m benchmark.research.supplementary_eval --expanded benchmark/dataset_expanded.jsonl` |
| Table 7 compound-request retrieval | `python -m benchmark.research.chains` |
| Sec. 7 static incentive (0.584, 19.6 %, 95.84 %) | `python -m benchmark.research.qos_incentive` |
| Sec. 7 dynamic market; Table 10 cold start | `python -m benchmark.research.qos_dynamic --experiment all` (and `--prior optimistic`) |
| Table 11 quality / cost / latency sweeps | `python -m benchmark.research.qos_sensitivity` (full grid: `benchmark/qos_sensitivity_results.json`) |
| Fig. 2 | `python -m benchmark.research.plot_qos_dynamic` |

**Code and complete per-call logs: re-score offline, re-run with credentials**

| Paper result | Saved log | Re-run |
|---|---|---|
| Table 6 selector and Jev rows (707 requests; per-request log) | `benchmark/supplementary_eval.json` | `OP=<openrouter key> python -m benchmark.research.supplementary_eval --expanded benchmark/dataset_expanded.jsonl --selector --jev` (Vertex AI + OpenRouter) |
| Table 5 Jev row (two passes) | `benchmark/jev_results.jsonl`, `benchmark/jev_results_openrouter.jsonl` | `OP=<openrouter key> python -m benchmark.jev_comparison` |
| Table 8 full prompt tokens, ROUTER vs. EAGER (3 requests x 3 runs x 2 arms) | `benchmark/table7_chain_cost.json` | `python -m benchmark.research.table7_chain_cost --spawn-servers --repeats 3` (Vertex AI; starts the 18 synthetic MCP servers locally) |

**Code only: re-run with Vertex AI, no per-query logs were saved**

| Paper result | Re-run |
|---|---|
| Table 4 selector rows (published run; a logged top-5 re-run is in `benchmark/supplementary_eval.json`, 98.22 % on the 505) | `python -m benchmark.research.recovery --k 3` and `--k 5` |
| Table 9 EAGER prompt control | `python -m benchmark.research.eager_control` (needs `google-adk`) |

### Table 8 harness (`table7_chain_cost.py`)

Both arms are Google ADK agents with the same model, instruction, and the 18
synthetic MCP servers in `synthetic_servers/`; only their tools differ. EAGER
gets all 101 operations as tools. ROUTER's primary agent gets one tool,
`route_and_execute`; behind it the orchestrator embeds the request, takes the
FAISS top 10, makes one planning call that orders the steps and fills their
arguments, and executes the plan. The ROUTER total counts every generation
prompt on that path: every primary-agent turn plus every planning call. The
log records each call's prompt tokens, the shortlist, the plan, the executed
operations, and the tool outputs. With `--spawn-servers` the harness points
every MCP client at the local servers (overriding any `*_MCP_URL` in a `.env`),
checks that all 18 servers answer a tool call before starting, and rejects and
retries any run in which a downstream tool call fails, so every recorded run
executed all of its tool calls.

### Rebuilding the vectors

The cached vectors were produced with `gemini-embedding-001` (768 dimensions,
`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types). To rebuild them, copy
`.env.example` to `.env`, point it at a Vertex AI project, delete `data/cache/`,
and run `python -m scripts.build_index` followed by any experiment; each
script re-embeds what it needs and caches it. A rebuild in October 2026
reproduced every table exactly (vectors matched the saved run to 3e-7), so the
model is deterministic at this setting.

## Query authorship

**Supplementary requests.** To probe sensitivity to query wording, two
model-authored, scenario-based requests per operation (202 in total, ids
`*_006` and `*_007`) were added. The authoring model had access to the
operation descriptions, schemas, indexed examples, and the original benchmark,
so they are a supplementary, non-independent stress test rather than a sample
of real user traffic. None duplicates or nearly duplicates (token Jaccard
>= 0.8) an indexed example or an original query.

**Original queries.**
The authors wrote the 505 queries after the registry was fixed, consulting
each operation's description and input schema and phrasing each query
differently from the registry's example strings. A post-hoc overlap check
found two queries that coincide with a positive example up to capitalization
(`email_search_001`, `git_diff_001`) and one that differs by a single word
(`uuid_generate_001`); all three are routed correctly by every method. The
benchmark is synthetic, English-only, balanced by construction, and reflects
the authors' phrasing choices; see `dataset/DATASET_CARD.md` for intended uses
and limitations.

## Headline numbers (n = 505)

| Method | Top-1 | 95 % Wilson CI |
|---|---:|---|
| BM25 | 62.97 % | 58.7 - 67.1 |
| RRF (BM25 + dense top-5) | 86.93 % | 83.7 - 89.6 |
| Flat dense (FAISS, one vector per operation) | 95.64 % | 93.5 - 97.1 |
| Class-gated hierarchy, beam 1 | 87.13 % | - |
| Multi-vector, 5-fold CV | 97.62 % | 95.9 - 98.6 |
| Jev 1.13, all 101 descriptions | 97.82 % | 96.1 - 98.8 |
| Selector over flat top 5 | 98.02 % | 96.4 - 98.9 |

On the 202 supplementary requests (Table 6), flat dense retrieval scores
94.06 %, the class-gated hierarchy 83.17 %, BM25 72.28 %, multi-vector 93.56 %
(its 505-query gain does not transfer), the shortlist selector 100 %, and Jev
98.51 %. Over all 707: flat 95.19 %, selector 98.73 %, Jev 98.02 %.

On three matched compound requests (Table 8), counting every prompt on both
paths, ROUTER used 5.3x fewer prompt tokens than an eager ADK agent loading all
101 tool schemas (pooled over nine runs; 4.0-9.4x per request). The advantage
is 8.9-9.4x when the primary agent delegates the whole request to the
orchestrator in one call and 3.9-4.7x when it splits the chain into per-step
calls.


## License

Code is released under the MIT License (`LICENSE`). The dataset files under
`dataset/`, `data/tools.json`, `data/providers.json`, and `benchmark/*.jsonl`
are released under CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`.
