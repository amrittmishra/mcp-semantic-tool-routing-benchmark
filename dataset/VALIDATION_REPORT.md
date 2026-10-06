# Validation Report

Status: **PASS**

Generated: 2026-08-11

- PASS: Server row count is exactly 18
- PASS: Tool row count is exactly 101
- PASS: Query row count is exactly 505
- PASS: Positive-example row count is exactly 303
- PASS: Negative-example row count is exactly 303
- PASS: Tool-parameter row count is exactly 160
- PASS: Server IDs are unique
- PASS: Operation IDs are unique
- PASS: Tool IDs are unique
- PASS: Query IDs are unique
- PASS: Query texts are unique
- PASS: Every tool server_id joins to servers.csv
- PASS: Every query operation label joins to tools.csv
- PASS: Every query tool label joins to tools.csv
- PASS: Every query server label joins to servers.csv
- PASS: Every query operation/tool/server label triple matches the canonical registry
- PASS: Every operation has exactly five benchmark queries
- PASS: All required tool fields are populated
- PASS: All required server fields are populated
- PASS: All required query fields are populated
- PASS: No released text cell begins with a spreadsheet-formula prefix
- PASS: Basic sensitive-content scan found no private keys, credential values, or non-placeholder email addresses

The sensitive-content scan is intentionally basic and does not replace a legal,
privacy, or institutional review before publication.
