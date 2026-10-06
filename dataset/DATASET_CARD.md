# Dataset Card

## Dataset summary

The MCP Semantic Tool Routing Benchmark evaluates whether a natural-language
request can be mapped to one of 101 semantic operations distributed across 18
synthetic MCP servers. It contains 505 hand-authored English queries with gold
operation, tool, and server labels.

## Intended uses

- Tool-retrieval and semantic-routing evaluation
- Flat versus hierarchical retrieval comparisons
- Tool-description representation and shortlist-selection research
- Reproducibility studies using the accompanying registry

## Out-of-scope uses

- Production safety or security certification
- End-to-end execution-quality claims
- Measuring real user behavior
- Training or evaluating demographic or cultural fairness
- Treating query order as a validated difficulty scale

## Data collection and annotation

The registry is synthetic. Five benchmark queries were authored for each
operation independently of the positive examples stored in the tool registry.
Gold tool and server labels were resolved from the canonical registry rather
than entered separately, preventing label drift.

## Personal and sensitive information

The package is derived from synthetic tool definitions and authored prompts, not
user logs. The release validation performs a basic scan for email addresses,
credentials, private keys, and formula-injection prefixes; this is not a legal or
privacy audit.

## Known limitations and biases

The dataset is small, English-only, balanced by construction, and limited to the
101 operations in the synthetic registry. Query phrasing and ambiguity reflect
the authors' choices. Results may not transfer to larger, multilingual,
adversarial, or highly redundant real-world tool catalogues.
