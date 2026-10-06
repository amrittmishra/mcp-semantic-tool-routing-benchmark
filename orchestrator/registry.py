"""Parse ``data/tools.json`` into flat capability records.

The registry is the *source of truth*. Everything under ``data/cache/`` is a
derived build artifact that can be regenerated from this file plus
``gemini-embedding-001``.

The central design decision of Branch 1 lives here: we flatten
``servers[].tools[]`` into one record per **semantic operation**, and the
downstream ``server_id`` is carried along only as execution metadata. There is
exactly one vector per operation -- never one vector per MCP server.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from orchestrator import config


@dataclass(frozen=True)
class ToolRecord:
    """One semantic operation. In Branch 1 this is also exactly one tool."""

    operation_id: str
    tool_id: str
    name: str
    description: str
    taxonomy_class: str
    taxonomy_type: str
    positive_examples: tuple[str, ...]
    negative_examples: tuple[str, ...]
    input_schema: dict[str, Any]

    # Execution metadata -- deliberately *not* weighted in the embedding doc.
    server_id: str
    server_name: str
    server_description: str

    @property
    def required_fields(self) -> list[str]:
        return [
            key
            for key, spec in self.input_schema.items()
            if spec.get("required", False)
        ]

    def json_schema(self) -> dict[str, Any]:
        """Registry schema -> a real JSON Schema object for the LLM/validation."""
        properties: dict[str, Any] = {}
        for key, spec in self.input_schema.items():
            prop = {"type": spec.get("type", "string")}
            if "description" in spec:
                prop["description"] = spec["description"]
            if "enum" in spec:
                prop["enum"] = spec["enum"]
            if spec.get("type") == "array":
                prop["items"] = spec.get("items", {"type": "string"})
            properties[key] = prop
        return {
            "type": "object",
            "properties": properties,
            "required": self.required_fields,
        }


@dataclass
class Registry:
    path: pathlib.Path
    raw: dict[str, Any]
    # One canonical record per operation. Under QoS these are the incumbents.
    tools: list[ToolRecord] = field(default_factory=list)
    # operation_id -> every implementation, incumbent first. In Branch 1 each
    # list has exactly one member.
    provider_index: dict[str, list[ToolRecord]] = field(default_factory=dict)

    @property
    def branch(self) -> str:
        return self.raw.get("branch", "unknown")

    @property
    def qos_enabled(self) -> bool:
        return bool(self.raw.get("qos_enabled", False))

    @property
    def servers(self) -> list[dict[str, Any]]:
        """Raw server entries. Execution metadata -- never a retrieval target."""
        return self.raw.get("servers", [])

    @property
    def server_ids(self) -> list[str]:
        return [s["server_id"] for s in self.servers]

    def hash(self) -> str:
        return registry_hash(self.raw)

    @property
    def all_records(self) -> list[ToolRecord]:
        """Every implementation, not just the canonical one per operation.

        ``tools`` holds one record per operation because that is what gets
        embedded. A downstream server needs every record it owns, including the
        competing implementations that share an operation with another server.
        """
        if not self.qos_enabled:
            return list(self.tools)
        return [record for records in self.provider_index.values() for record in records]

    def tools_for_server(self, server_id: str) -> list[ToolRecord]:
        return [t for t in self.all_records if t.server_id == server_id]

    def record_by_tool_id(self, tool_id: str) -> ToolRecord | None:
        """Look up any implementation by its tool_id, canonical or competing."""
        return next((t for t in self.all_records if t.tool_id == tool_id), None)

    def by_tool_id(self, tool_id: str) -> ToolRecord | None:
        return next((t for t in self.tools if t.tool_id == tool_id), None)

    def by_operation_id(self, operation_id: str) -> ToolRecord | None:
        """The canonical record for an operation.

        Under QoS the operation has several implementations; this returns the
        incumbent, whose description and schema are the shared ones. Retrieval
        and argument extraction both use this record, which is why the index
        stays at one vector per operation however many providers exist.
        """
        return next((t for t in self.tools if t.operation_id == operation_id), None)

    def providers_for(self, operation_id: str) -> list[ToolRecord]:
        """Every implementation of an operation, incumbent first."""
        return self.provider_index.get(operation_id, [])

    @property
    def contested_operations(self) -> list[str]:
        return sorted(op for op, ps in self.provider_index.items() if len(ps) > 1)


def registry_hash(raw: dict[str, Any]) -> str:
    """Deterministic SHA-256 over the canonical registry contents.

    Canonical = sorted keys, no insignificant whitespace, so reformatting
    ``tools.json`` does not spuriously invalidate a cached index, but any
    semantic change to it does.
    """
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def load_registry(path: pathlib.Path | str | None = None) -> Registry:
    path = pathlib.Path(path) if path is not None else config.REGISTRY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Tool registry not found at {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))
    registry = Registry(path=path, raw=raw)
    qos_mode = bool(raw.get("qos_enabled", False))

    seen_operations: dict[str, str] = {}
    seen_tool_ids: dict[str, str] = {}

    for server in raw.get("servers", []):
        for tool in server.get("tools", []):
            operation_id = tool["operation_id"]
            tool_id = tool["tool_id"]

            # Branch 1 invariant: exactly one implementation per operation.
            # A registry that declares qos_enabled opts into Branch 2, where
            # duplicates are the point and get grouped into a provider set.
            if operation_id in seen_operations and not qos_mode:
                raise ValueError(
                    f"Duplicate operation_id {operation_id!r}: already implemented by "
                    f"{seen_operations[operation_id]!r}. Branch 1 (taxonomy-router) "
                    "allows exactly one implementation per operation; multiple "
                    "providers belong in Branch 2 (QoS)."
                )
            if tool_id in seen_tool_ids:
                raise ValueError(
                    f"Duplicate tool_id {tool_id!r} in servers "
                    f"{seen_tool_ids[tool_id]!r} and {server['server_id']!r}."
                )
            seen_operations[operation_id] = server["server_id"]
            seen_tool_ids[tool_id] = server["server_id"]

            taxonomy = tool.get("taxonomy", {})
            record = ToolRecord(
                    operation_id=operation_id,
                    tool_id=tool_id,
                    name=tool.get("name", tool_id),
                    description=tool.get("description", ""),
                    taxonomy_class=taxonomy.get("class", ""),
                    taxonomy_type=taxonomy.get("type", ""),
                    positive_examples=tuple(tool.get("positive_examples", [])),
                    negative_examples=tuple(tool.get("negative_examples", [])),
                    input_schema=tool.get("input_schema", {}),
                    server_id=server["server_id"],
                    server_name=server.get("name", server["server_id"]),
                    server_description=server.get("description", ""),
                )

            # The first implementation seen is the incumbent and becomes the
            # canonical record: the one that is embedded and whose schema drives
            # argument extraction. Later ones join its provider set.
            bucket = registry.provider_index.setdefault(operation_id, [])
            if not bucket:
                registry.tools.append(record)
            bucket.append(record)

    return registry


def select_tools(tools: list[ToolRecord], limit: int | None) -> list[ToolRecord]:
    """Deterministically pick a subset of ``limit`` tools for scaling runs.

    Round-robin across servers in registry order, so a 10-tool index still
    covers ten different domains rather than the first two servers. Purely a
    function of the registry order -- same registry, same subset, every time.
    """
    if limit is None or limit >= len(tools):
        return list(tools)
    if limit <= 0:
        raise ValueError("--tool-limit must be positive")

    by_server: dict[str, list[ToolRecord]] = {}
    for tool in tools:
        by_server.setdefault(tool.server_id, []).append(tool)

    selected: list[ToolRecord] = []
    depth = 0
    while len(selected) < limit:
        added_this_round = False
        for bucket in by_server.values():
            if depth < len(bucket):
                selected.append(bucket[depth])
                added_this_round = True
                if len(selected) == limit:
                    break
        if not added_this_round:
            break
        depth += 1

    # Restore registry order so the FAISS insertion order stays stable.
    order = {tool.operation_id: i for i, tool in enumerate(tools)}
    return sorted(selected, key=lambda t: order[t.operation_id])


def embedding_document(tool: ToolRecord) -> str:
    """Build the text that gets embedded for one capability.

    Weighted toward *semantics* -- name, taxonomy, purpose, and example
    phrasings. ``server_id`` and provider names are intentionally absent: they
    are execution metadata, and including them would pull retrieval toward
    "which vendor" instead of "which capability".
    """
    lines = [
        f"Capability: {tool.name}",
        "",
        "Taxonomy:",
        f"{tool.taxonomy_class} > {tool.taxonomy_type} > {tool.operation_id}",
        "",
        "Purpose:",
        tool.description,
    ]

    if tool.positive_examples:
        lines += ["", "Typical user requests:"]
        lines += [f"- {example}" for example in tool.positive_examples]

    if tool.negative_examples:
        lines += ["", "Do not use this capability for:"]
        lines += [f"- {example}" for example in tool.negative_examples]

    return "\n".join(lines)
