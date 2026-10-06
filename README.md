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
| `benchmark/results.jsonl` | Saved flat-FAISS run: top-5 candidates and scores per query (Table 1 baseline, 483/505) |
| `benchmark/jev_results.jsonl`, `jev_results_openrouter.jsonl` | Two independent Jev 1.13 passes over the 505 queries (494/505 each) |
| `benchmark/qos_dynamic_results.json`, `qos_dynamic_optimistic.json` | Saved dynamic QoS simulation outputs |
| `orchestrator/` | The minimal library the scripts import: registry loader, embedding document builder, FAISS index, router, Jev client, QoS registry and simulator |
| `benchmark/`, `benchmark/research/` | Experiment scripts (table below) |

## Reproducing the paper's tables

Python 3.11+. Install and run from the repository root:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

No credentials are needed for anything marked *offline*.

| Paper table / result | Command | Needs |
|---|---|---|
| Table 1 flat vs. hierarchical routing; Table 2 authored vs. k-means gates | `python -m benchmark.hierarchy_experiment` | offline |
| Table 3 accuracy vs. catalogue size | `python -m benchmark.research.scaling` | offline |
| Table 4 mean-centering / ABTT / CSLS rows | `python -m benchmark.research.reranking` | offline |
| Table 4 multi-vector 2-fold and 5-fold rows | `python -m benchmark.research.tune_blend` | offline |
| Multi-vector ablations (Sec. 5.2) | `python -m benchmark.research.multivector` | offline |
| Table 5 BM25, RRF, interpolation | `python -m benchmark.research.lexical_hybrid` (add `--dense-matrix` for full-ranking variants, see docstring) | offline |
| Table 5 Jev row | `OP=<openrouter key> python -m benchmark.jev_comparison` | OpenRouter |
| Table 4 selector rows | `python -m benchmark.research.recovery --k 5` | Vertex AI (Gemini) |
| Table 6 compound-request retrieval | `python -m benchmark.research.chains` | offline |
| Table 8 EAGER prompt control | `python -m benchmark.research.eager_control` | Vertex AI + `google-adk` |
| Sec. 7 static QoS incentive (0.584, 19.6 %, 95.84 %) | `python -m benchmark.research.qos_incentive` | offline |
| Sec. 7 / Table 9 dynamic QoS, cold start | `python -m benchmark.research.qos_dynamic --experiment all` and `--prior optimistic`; `--experiment cold-start` | offline |
| Fig. 2 | `python -m benchmark.research.plot_qos_dynamic` | offline (`matplotlib`) |

The three matched ROUTER-vs-EAGER chain loops (Table 7) were measured against
the live orchestrator and its 18 synthetic MCP servers in the full system
repository and are not reproducible from this package alone; their raw token
counts are reported in the paper.

### Rebuilding the vectors

The cached vectors were produced with `gemini-embedding-001` (768 dimensions,
`RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types). To rebuild them, copy
`.env.example` to `.env`, point it at a Vertex AI project, delete `data/cache/`,
and run `python -m scripts.build_index` followed by any experiment; each
script re-embeds what it needs and caches it. A rebuild in October 2026
reproduced every table exactly (vectors matched the saved run to 3e-7), so the
model is deterministic at this setting.

## Query authorship

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

## License

Code is released under the MIT License (`LICENSE`). The dataset files under
`dataset/`, `data/tools.json`, `data/providers.json`, and `benchmark/*.jsonl`
are released under CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`.
