# MCP Semantic Tool Routing Benchmark

Version 1.0.0 (2026-08-11)

This release packages the synthetic MCP registry and routing benchmark used by
the accompanying study into flat, UTF-8, RFC 4180-compatible CSV files.

## Contents

| File | Rows | Description |
|---|---:|---|
| `servers.csv` | 18 | One row per downstream synthetic MCP server. |
| `tools.csv` | 101 | One row per semantic operation. In this release, one operation maps to one tool and one server. |
| `queries.csv` | 505 | One row per hand-authored routing query with gold operation, tool, and server labels. |
| `tool_examples.csv` | 606 | Positive and negative examples used in tool descriptions, one example per row. |
| `tool_parameters.csv` | 160 | Flattened tool input schemas, one input parameter per row. |
| `data_dictionary.csv` | 39 | Machine-readable definitions for every released CSV column. |

An Excel convenience copy is provided as
`mcp_tool_routing_dataset_v1.0.0.xlsx`. The CSV files are the recommended
machine-readable release artifacts.

## Key joins

- `queries.expected_operation_id -> tools.operation_id`
- `queries.expected_tool_id -> tools.tool_id`
- `queries.expected_server_id -> servers.server_id`
- `tools.server_id -> servers.server_id`
- `tool_examples.operation_id -> tools.operation_id`
- `tool_parameters.operation_id -> tools.operation_id`

## Provenance

- Tool registry source: `data/tools.json`
- Query source: `benchmark/dataset.jsonl`, materialized from
  `benchmark/queries.py`
- The 505 English queries were hand-authored independently of the registry's
  positive-example field. They are not production traffic or user logs.
- The 18 MCP servers and their execution outputs are synthetic.
- `query_variant` preserves authoring order within an operation and must not be
  interpreted as a validated difficulty category.

## Release summary

- 18 servers
- 101 tools and 101 semantic operations
- 505 unique benchmark queries, exactly five per operation
- 303 positive and 303 negative tool-description examples
- 160 flattened input parameters
- 12 taxonomy classes and 38 class/type nodes

## Validation

The release validation checks unique identifiers, one-to-one operation/tool/server
labels, complete foreign-key joins, per-operation query balance, missing required
fields, duplicate query text, and CSV formula-injection prefixes. File hashes are
listed in `checksums.sha256` and `release_manifest.csv`.

## Limitations

This is a synthetic, English-language routing benchmark with five queries per
operation. It does not measure argument extraction, downstream execution,
adversarial tool descriptions, real-world task success, or demographic fairness.
The registry's one-operation/one-tool/one-server mapping reflects the evaluated
branch and should not be generalized to provider marketplaces.

## License

No license has been applied automatically. Select and approve a dataset license
with all contributors before uploading this package publicly. See
`LICENSE_PENDING.md`.

## Citation

Complete the repository URL and, when available, DOI fields in `CITATION.cff`
before publication.
