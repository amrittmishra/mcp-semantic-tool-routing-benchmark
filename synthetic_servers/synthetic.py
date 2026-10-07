"""Deterministic fake execution for every operation in the registry.

``synthetic_result(server_id, tool_id, arguments)`` is the single entry point.
The same inputs always produce the same output: any pseudo-randomness is seeded
from a SHA-256 of the canonicalized call, never from the clock or ``os.urandom``.

Nothing in here performs a real external action. ``read_file("/etc/passwd")``
returns a made-up string; ``send_email(...)`` returns a made-up message id. The
only real computation is pure arithmetic and encoding (calculator, statistics,
hashing, base64, date math) where a real answer is both harmless and more
useful for demonstrating end-to-end argument extraction.
"""

from __future__ import annotations

import ast
import base64
import binascii
import datetime as dt
import hashlib
import json
import math
import operator
import random
import statistics
import uuid
from typing import Any, Callable

UUID_NAMESPACE = uuid.UUID("6f2c1e2a-2b7d-5f6a-9c31-0a1b2c3d4e5f")

# --- deterministic helpers --------------------------------------------------

_FIRST_NAMES = ["Alice", "Bob", "Carla", "Dev", "Elena", "Farid", "Gita", "Hugo"]
_DOMAINS = ["example.com", "example.org", "synthetic.test"]
_BRANCHES = ["main", "develop", "feature/router", "release/1.0", "hotfix/auth"]


def _canonical(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)


def _seed(server_id: str, tool_id: str, arguments: dict[str, Any]) -> int:
    digest = hashlib.sha256(f"{server_id}|{tool_id}|{_canonical(arguments)}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(server_id: str, tool_id: str, arguments: dict[str, Any]) -> random.Random:
    return random.Random(_seed(server_id, tool_id, arguments))


def _fake_id(prefix: str, rng: random.Random) -> str:
    return f"{prefix}_{rng.getrandbits(40):010x}"


def _fake_person(rng: random.Random) -> dict[str, str]:
    name = rng.choice(_FIRST_NAMES)
    return {"name": name, "email": f"{name.lower()}@{rng.choice(_DOMAINS)}"}


def _fake_timestamp(rng: random.Random, day_offset: int = 0) -> str:
    """A stable timestamp anchored to a fixed epoch, never to 'now'."""
    base = dt.datetime(2026, 1, 5, 9, 0, tzinfo=dt.timezone.utc)
    delta = dt.timedelta(days=day_offset + rng.randrange(0, 120), hours=rng.randrange(0, 9))
    return (base + delta).isoformat()


def _parse_datetime(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    for parser in (dt.datetime.fromisoformat,):
        try:
            return parser(text)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%d %H:%M"):
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


# --- safe arithmetic --------------------------------------------------------

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": lambda: math.pi,
}


def safe_eval(expression: str) -> float:
    """Evaluate a pure arithmetic expression via AST -- never ``eval``."""

    def visit(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](visit(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ValueError(f"unsupported function {node.func.id!r}")
            return fn(*[visit(arg) for arg in node.args])
        if isinstance(node, ast.Name) and node.id in ("pi", "e"):
            return math.pi if node.id == "pi" else math.e
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    return visit(ast.parse(expression.strip(), mode="eval"))


# --- per-operation handlers -------------------------------------------------
# Signature: (arguments, rng) -> result payload

Handler = Callable[[dict[str, Any], random.Random], dict[str, Any]]


def _numbers(values: Any) -> list[float]:
    out: list[float] = []
    if isinstance(values, str):
        values = [v for v in values.replace(",", " ").split() if v]
    for value in values or []:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


HANDLERS: dict[str, Handler] = {
    # --- filesystem ---
    "FILE_READ": lambda a, r: {
        "path": a.get("path"),
        "content": f"Synthetic file contents for {a.get('path')}\nline 2\nline 3",
        "bytes": 64 + r.randrange(0, 4096),
        "encoding": "utf-8",
    },
    "FILE_WRITE": lambda a, r: {
        "path": a.get("path"),
        "bytes_written": len(a.get("content", "") or ""),
        "created": True,
        "note": "No file was written. Synthetic server.",
    },
    "FILE_APPEND": lambda a, r: {
        "path": a.get("path"),
        "bytes_appended": len(a.get("content", "") or ""),
        "note": "No file was modified. Synthetic server.",
    },
    "FILE_COPY": lambda a, r: {
        "source_path": a.get("source_path"),
        "destination_path": a.get("destination_path"),
        "copied": True,
    },
    "FILE_MOVE": lambda a, r: {
        "source_path": a.get("source_path"),
        "destination_path": a.get("destination_path"),
        "moved": True,
    },
    "FILE_DELETE": lambda a, r: {
        "path": a.get("path"),
        "deleted": True,
        "note": "Nothing was deleted. Synthetic server.",
    },
    "DIRECTORY_LIST": lambda a, r: {
        "path": a.get("path"),
        "entries": [
            {"name": "README.md", "type": "file", "bytes": 1024 + r.randrange(0, 512)},
            {"name": "notes.txt", "type": "file", "bytes": 256 + r.randrange(0, 512)},
            {"name": "reports", "type": "directory", "bytes": 0},
        ],
    },
    "FILE_METADATA": lambda a, r: {
        "path": a.get("path"),
        "bytes": 2048 + r.randrange(0, 8192),
        "modified_at": _fake_timestamp(r),
        "mode": "0644",
        "owner": "synthetic",
    },
    # --- documents ---
    "PDF_TEXT_EXTRACT": lambda a, r: {
        "path": a.get("path"),
        "pages": 3 + r.randrange(0, 20),
        "text": f"Synthetic extracted PDF text from {a.get('path')}.",
    },
    "DOCX_TEXT_EXTRACT": lambda a, r: {
        "path": a.get("path"),
        "paragraphs": 8 + r.randrange(0, 40),
        "text": f"Synthetic extracted DOCX text from {a.get('path')}.",
    },
    "DOCUMENT_SUMMARIZE": lambda a, r: {
        "summary": "Synthetic summary: the document covers three main points.",
        "key_points": ["Synthetic point one", "Synthetic point two", "Synthetic point three"],
    },
    "DOCUMENT_SEARCH": lambda a, r: {
        "query": a.get("query"),
        "matches": [
            {"page": 1 + r.randrange(0, 12), "excerpt": f"...synthetic match for {a.get('query')}..."},
            {"page": 1 + r.randrange(0, 12), "excerpt": "...another synthetic match..."},
        ],
    },
    "DOCUMENT_SECTION_EXTRACT": lambda a, r: {
        "section": a.get("section"),
        "content": f"Synthetic contents of section {a.get('section')!r}.",
    },
    # --- database ---
    "SQL_QUERY": lambda a, r: {
        "query": a.get("query"),
        "columns": ["id", "name"],
        "rows": [[1, "Alice"], [2, "Bob"], [3, "Carla"]],
        "row_count": 3,
    },
    "DATABASE_SCHEMA_INSPECT": lambda a, r: {
        "tables": [
            {"name": "users", "columns": ["id", "name", "email", "created_at"]},
            {"name": "orders", "columns": ["id", "user_id", "total", "status"]},
        ]
    },
    "DATABASE_INSERT": lambda a, r: {
        "table": a.get("table"),
        "inserted_id": 1000 + r.randrange(0, 9000),
        "rows_affected": 1,
        "note": "No database was written. Synthetic server.",
    },
    "DATABASE_UPDATE": lambda a, r: {
        "table": a.get("table"),
        "rows_affected": r.randrange(1, 5),
        "note": "No database was written. Synthetic server.",
    },
    "DATABASE_DELETE": lambda a, r: {
        "table": a.get("table"),
        "rows_affected": r.randrange(1, 4),
        "note": "No database was written. Synthetic server.",
    },
    "DATABASE_AGGREGATE": lambda a, r: {
        "table": a.get("table"),
        "aggregation": a.get("aggregation"),
        "group_by": a.get("group_by"),
        "results": [
            {"group": "alpha", "value": round(r.uniform(10, 500), 2)},
            {"group": "beta", "value": round(r.uniform(10, 500), 2)},
        ],
    },
    # --- web ---
    "WEB_FETCH": lambda a, r: {
        "url": a.get("url"),
        "status": 200,
        "content_type": "text/html",
        "body": f"<html><body>Synthetic page for {a.get('url')}</body></html>",
        "note": "No HTTP request was made. Synthetic server.",
    },
    "WEB_TEXT_EXTRACT": lambda a, r: {
        "url": a.get("url"),
        "text": f"Synthetic readable text extracted from {a.get('url')}.",
        "word_count": 120 + r.randrange(0, 800),
    },
    "WEB_LINK_EXTRACT": lambda a, r: {
        "url": a.get("url"),
        "links": [
            {"text": "About", "href": "https://example.com/about"},
            {"text": "Pricing", "href": "https://example.com/pricing"},
            {"text": "Docs", "href": "https://example.com/docs"},
        ],
    },
    "WEB_METADATA": lambda a, r: {
        "url": a.get("url"),
        "title": "Synthetic Page Title",
        "description": "Synthetic meta description.",
        "og:image": "https://example.com/synthetic.png",
    },
    "WEB_STRUCTURED_EXTRACT": lambda a, r: {
        "url": a.get("url"),
        "fields": a.get("fields", []),
        "records": [
            {field: f"synthetic-{field}-{i}" for field in (a.get("fields") or ["value"])}
            for i in range(1, 4)
        ],
    },
    "WEB_SEARCH": lambda a, r: {
        "query": a.get("query"),
        "results": [
            {
                "title": f"Synthetic result {i} for {a.get('query')}",
                "url": f"https://example.com/result/{i}",
                "snippet": "Synthetic snippet text.",
            }
            for i in range(1, 4)
        ],
    },
    # --- research ---
    "PAPER_SEARCH": lambda a, r: {
        "query": a.get("query"),
        "papers": [
            {
                "paper_id": _fake_id("arxiv", r),
                "title": f"Synthetic Paper {i} on {a.get('query')}",
                "authors": [_fake_person(r)["name"], _fake_person(r)["name"]],
                "year": 2020 + r.randrange(0, 6),
            }
            for i in range(1, 4)
        ],
    },
    "PAPER_METADATA": lambda a, r: {
        "paper_id": a.get("paper_id"),
        "title": "Synthetic Paper Title",
        "venue": "Synthetic Conference on Routing",
        "year": 2020 + r.randrange(0, 6),
        "doi": f"10.0000/synthetic.{r.getrandbits(20)}",
    },
    "PAPER_CITATIONS": lambda a, r: {
        "paper_id": a.get("paper_id"),
        "citation_count": r.randrange(3, 400),
        "citations": [
            {"paper_id": _fake_id("arxiv", r), "title": f"Synthetic Citing Work {i}"}
            for i in range(1, 4)
        ],
    },
    "PAPER_SUMMARIZE": lambda a, r: {
        "summary": "Synthetic summary: the paper proposes a method and evaluates it.",
        "contributions": ["Synthetic contribution A", "Synthetic contribution B"],
    },
    "PAPER_COMPARE": lambda a, r: {
        "papers": a.get("papers", []),
        "comparison": [
            {"dimension": "method", "finding": "Synthetic methodological difference."},
            {"dimension": "dataset", "finding": "Synthetic dataset difference."},
        ],
    },
    # --- email ---
    "EMAIL_SEND": lambda a, r: {
        "message_id": _fake_id("msg", r),
        "to": a.get("to", []),
        "subject": a.get("subject"),
        "delivered": True,
        "note": "No email was sent. Synthetic server.",
    },
    "EMAIL_SEARCH": lambda a, r: {
        "query": a.get("query"),
        "matches": [
            {
                "message_id": _fake_id("msg", r),
                "subject": "Synthetic result",
                "sender": "alice@example.com",
                "received_at": _fake_timestamp(r),
            },
            {
                "message_id": _fake_id("msg", r),
                "subject": "Synthetic follow-up",
                "sender": "bob@example.org",
                "received_at": _fake_timestamp(r),
            },
        ],
    },
    "EMAIL_READ": lambda a, r: {
        "message_id": a.get("message_id"),
        "subject": "Synthetic subject",
        "sender": "alice@example.com",
        "body": "Synthetic email body.",
    },
    "EMAIL_REPLY": lambda a, r: {
        "message_id": a.get("message_id"),
        "reply_id": _fake_id("msg", r),
        "sent": True,
        "note": "No reply was sent. Synthetic server.",
    },
    "EMAIL_DRAFT": lambda a, r: {
        "draft_id": _fake_id("draft", r),
        "to": a.get("to", []),
        "subject": a.get("subject"),
        "saved": True,
    },
    "EMAIL_ARCHIVE": lambda a, r: {
        "message_id": a.get("message_id"),
        "archived": True,
        "note": "No mailbox was modified. Synthetic server.",
    },
    # --- calendar ---
    "CALENDAR_CREATE": lambda a, r: {
        "event_id": _fake_id("evt", r),
        "title": a.get("title"),
        "start_time": a.get("start_time"),
        "end_time": a.get("end_time"),
        "created": True,
        "note": "No calendar was modified. Synthetic server.",
    },
    "CALENDAR_SEARCH": lambda a, r: {
        "query": a.get("query"),
        "events": [
            {
                "event_id": _fake_id("evt", r),
                "title": "Synthetic standup",
                "start_time": _fake_timestamp(r),
            },
            {
                "event_id": _fake_id("evt", r),
                "title": "Synthetic design review",
                "start_time": _fake_timestamp(r),
            },
        ],
    },
    "CALENDAR_UPDATE": lambda a, r: {
        "event_id": a.get("event_id"),
        "updates": a.get("updates", {}),
        "updated": True,
    },
    "CALENDAR_DELETE": lambda a, r: {
        "event_id": a.get("event_id"),
        "deleted": True,
        "note": "No calendar was modified. Synthetic server.",
    },
    "CALENDAR_AVAILABILITY": lambda a, r: {
        "start_time": a.get("start_time"),
        "end_time": a.get("end_time"),
        "available": bool(r.getrandbits(1)),
        "conflicts": [{"event_id": _fake_id("evt", r), "title": "Synthetic conflict"}],
    },
    # --- messaging ---
    "MESSAGE_SEND": lambda a, r: {
        "channel": a.get("channel"),
        "message_id": _fake_id("m", r),
        "sent": True,
        "note": "No message was sent. Synthetic server.",
    },
    "MESSAGE_SEARCH": lambda a, r: {
        "query": a.get("query"),
        "matches": [
            {
                "message_id": _fake_id("m", r),
                "channel": "#general",
                "author": _fake_person(r)["name"],
                "text": f"Synthetic message mentioning {a.get('query')}",
            }
        ],
    },
    "MESSAGE_THREAD_REPLY": lambda a, r: {
        "thread_id": a.get("thread_id"),
        "message_id": _fake_id("m", r),
        "sent": True,
    },
    "CHANNEL_LIST": lambda a, r: {
        "channels": [
            {"name": "#general", "members": 40 + r.randrange(0, 60)},
            {"name": "#engineering", "members": 10 + r.randrange(0, 40)},
            {"name": "#random", "members": 5 + r.randrange(0, 30)},
        ]
    },
    "CHANNEL_HISTORY": lambda a, r: {
        "channel": a.get("channel"),
        "messages": [
            {
                "message_id": _fake_id("m", r),
                "author": _fake_person(r)["name"],
                "text": f"Synthetic history message {i}",
                "sent_at": _fake_timestamp(r),
            }
            for i in range(1, 4)
        ],
    },
    # --- git ---
    "GIT_STATUS": lambda a, r: {
        "repo_path": a.get("repo_path"),
        "branch": r.choice(_BRANCHES),
        "staged": ["orchestrator/router.py"],
        "modified": ["README.md"],
        "untracked": ["scratch.txt"],
    },
    "GIT_DIFF": lambda a, r: {
        "repo_path": a.get("repo_path"),
        "base": a.get("base", "HEAD~1"),
        "target": a.get("target", "HEAD"),
        "files_changed": 2,
        "diff": (
            "--- a/README.md\n+++ b/README.md\n"
            "@@ -1,3 +1,4 @@\n synthetic diff\n+added synthetic line\n"
        ),
    },
    "GIT_COMMIT_HISTORY": lambda a, r: {
        "repo_path": a.get("repo_path"),
        "commits": [
            {
                "sha": f"{r.getrandbits(160):040x}"[:40],
                "author": _fake_person(r)["name"],
                "message": f"Synthetic commit {i}",
                "date": _fake_timestamp(r),
            }
            for i in range(1, int(a.get("limit") or 3) + 1)
        ][:20],
    },
    "GIT_BRANCH_LIST": lambda a, r: {
        "repo_path": a.get("repo_path"),
        "branches": _BRANCHES,
        "current": r.choice(_BRANCHES),
    },
    "GIT_CREATE_BRANCH": lambda a, r: {
        "repo_path": a.get("repo_path"),
        "branch_name": a.get("branch_name"),
        "base": a.get("base", "main"),
        "created": True,
        "note": "No repository was modified. Synthetic server.",
    },
    "GIT_COMMIT": lambda a, r: {
        "repo_path": a.get("repo_path"),
        "sha": f"{r.getrandbits(160):040x}"[:40],
        "message": a.get("message"),
        "committed": True,
        "note": "No repository was modified. Synthetic server.",
    },
    # --- issues ---
    "ISSUE_SEARCH": lambda a, r: {
        "query": a.get("query"),
        "issues": [
            {
                "issue_id": f"ISSUE-{100 + r.randrange(0, 900)}",
                "title": f"Synthetic issue about {a.get('query')}",
                "state": r.choice(["open", "closed"]),
            }
            for _ in range(3)
        ],
    },
    "ISSUE_READ": lambda a, r: {
        "issue_id": a.get("issue_id"),
        "title": "Synthetic issue title",
        "state": "open",
        "body": "Synthetic issue body.",
        "assignee": _fake_person(r)["name"],
    },
    "ISSUE_CREATE": lambda a, r: {
        "issue_id": f"ISSUE-{100 + r.randrange(0, 900)}",
        "title": a.get("title"),
        "created": True,
        "note": "No tracker was modified. Synthetic server.",
    },
    "ISSUE_UPDATE": lambda a, r: {
        "issue_id": a.get("issue_id"),
        "updates": a.get("updates", {}),
        "updated": True,
    },
    "ISSUE_COMMENT": lambda a, r: {
        "issue_id": a.get("issue_id"),
        "comment_id": _fake_id("cmt", r),
        "posted": True,
    },
    # --- text ---
    "TEXT_SUMMARIZE": lambda a, r: {
        "summary": "Synthetic summary of the supplied text.",
        "input_characters": len(a.get("text", "") or ""),
    },
    "TEXT_TRANSLATE": lambda a, r: {
        "target_language": a.get("target_language"),
        "translated_text": (
            f"[synthetic {a.get('target_language')} translation] {a.get('text', '')}"
        ),
    },
    "TEXT_SENTIMENT": lambda a, r: {
        "label": r.choice(["positive", "neutral", "negative"]),
        "score": round(r.uniform(0.5, 0.99), 3),
    },
    "TEXT_KEYWORDS": lambda a, r: {
        "keywords": ["synthetic", "routing", "taxonomy", "capability"][: 2 + r.randrange(0, 3)]
    },
    "TEXT_ENTITIES": lambda a, r: {
        "entities": [
            {"text": "Alice", "type": "PERSON"},
            {"text": "Bangalore", "type": "LOCATION"},
            {"text": "2026-01-05", "type": "DATE"},
        ]
    },
    "TEXT_REWRITE": lambda a, r: {
        "instruction": a.get("instruction"),
        "rewritten_text": f"[synthetic rewrite: {a.get('instruction')}] {a.get('text', '')}",
    },
    "TEXT_CLASSIFY": lambda a, r: {
        "categories": a.get("categories", []),
        "label": (a.get("categories") or ["synthetic"])[0],
        "confidence": round(r.uniform(0.6, 0.99), 3),
    },
    # --- math (real deterministic arithmetic; no side effects) ---
    "CALCULATOR": lambda a, r: _calculator(a),
    "STATISTICS_DESCRIBE": lambda a, r: _describe(a),
    "PERCENTAGE_CHANGE": lambda a, r: _percentage_change(a),
    "COMPOUND_INTEREST": lambda a, r: _compound_interest(a),
    "LINEAR_EQUATION_SOLVE": lambda a, r: _solve_linear(a),
    # --- data analysis ---
    "DATASET_INSPECT": lambda a, r: {
        "dataset": a.get("dataset"),
        "rows": 500 + r.randrange(0, 5000),
        "columns": [
            {"name": "id", "dtype": "int64"},
            {"name": "name", "dtype": "object"},
            {"name": "amount", "dtype": "float64"},
        ],
    },
    "DATASET_FILTER": lambda a, r: {
        "dataset": a.get("dataset"),
        "condition": a.get("condition"),
        "rows_before": 1000,
        "rows_after": 100 + r.randrange(0, 800),
    },
    "DATASET_SORT": lambda a, r: {
        "dataset": a.get("dataset"),
        "columns": a.get("columns", []),
        "preview": [[1, "Alice", 10.5], [2, "Bob", 22.0]],
    },
    "DATASET_GROUP": lambda a, r: {
        "dataset": a.get("dataset"),
        "group_by": a.get("group_by", []),
        "aggregations": a.get("aggregations", {}),
        "groups": [
            {"key": "alpha", "value": round(r.uniform(1, 100), 2)},
            {"key": "beta", "value": round(r.uniform(1, 100), 2)},
        ],
    },
    "DATASET_COLUMN_CREATE": lambda a, r: {
        "dataset": a.get("dataset"),
        "column_name": a.get("column_name"),
        "expression": a.get("expression"),
        "created": True,
    },
    "DATASET_MISSING_VALUES": lambda a, r: {
        "dataset": a.get("dataset"),
        "missing": [
            {"column": "name", "missing": r.randrange(0, 20)},
            {"column": "amount", "missing": r.randrange(0, 50)},
        ],
    },
    # --- json (real, pure) ---
    "JSON_PARSE": lambda a, r: _json_parse(a),
    "JSON_QUERY": lambda a, r: _json_query(a),
    "JSON_VALIDATE": lambda a, r: _json_validate(a),
    "JSON_TRANSFORM": lambda a, r: {
        "transformation": a.get("transformation", {}),
        "output": {**(a.get("data") or {}), "_synthetic_transform": True},
    },
    "JSON_COMPARE": lambda a, r: _json_compare(a),
    # --- system (all fabricated; nothing is inspected) ---
    "SYSTEM_INFO": lambda a, r: {
        "os": "SyntheticOS 1.0",
        "kernel": "synthetic-5.0.0",
        "architecture": "x86_64",
        "hostname": "synthetic-host",
        "note": "No host was inspected. Synthetic server.",
    },
    "DISK_USAGE": lambda a, r: {
        "path": a.get("path", "/"),
        "total_gb": 512,
        "used_gb": 100 + r.randrange(0, 300),
        "note": "No host was inspected. Synthetic server.",
    },
    "MEMORY_USAGE": lambda a, r: {
        "total_mb": 32768,
        "used_mb": 4096 + r.randrange(0, 16384),
        "note": "No host was inspected. Synthetic server.",
    },
    "PROCESS_LIST": lambda a, r: {
        "processes": [
            {"pid": 1000 + r.randrange(0, 9000), "name": "synthetic-daemon", "cpu": 0.4},
            {"pid": 1000 + r.randrange(0, 9000), "name": "synthetic-worker", "cpu": 1.2},
        ],
        "note": "No host was inspected. Synthetic server.",
    },
    "ENVIRONMENT_READ": lambda a, r: {
        "name": a.get("name"),
        "value": f"synthetic-value-for-{a.get('name')}",
        "note": "The real process environment was NOT read. Synthetic server.",
    },
    # --- geo ---
    "GEOCODE": lambda a, r: {
        "location": a.get("location"),
        "latitude": round(r.uniform(-60, 60), 6),
        "longitude": round(r.uniform(-180, 180), 6),
        "confidence": round(r.uniform(0.7, 0.99), 3),
    },
    "REVERSE_GEOCODE": lambda a, r: {
        "latitude": a.get("latitude"),
        "longitude": a.get("longitude"),
        "address": "1 Synthetic Street, Example City, Exampleland",
    },
    "GEO_DISTANCE": lambda a, r: {
        "origin": a.get("origin"),
        "destination": a.get("destination"),
        "distance_km": round(r.uniform(5, 12000), 2),
        "method": "synthetic great-circle",
    },
    "TIMEZONE_LOOKUP": lambda a, r: {
        "location": a.get("location"),
        "timezone": r.choice(["Asia/Kolkata", "Europe/Berlin", "America/New_York"]),
        "utc_offset": r.choice(["+05:30", "+01:00", "-05:00"]),
    },
    "COUNTRY_LOOKUP": lambda a, r: {
        "country": a.get("country"),
        "capital": "Synthetic City",
        "population": 1_000_000 * r.randrange(1, 200),
        "currency": "SYN",
    },
    # --- media ---
    "IMAGE_METADATA": lambda a, r: {
        "path": a.get("path"),
        "width": r.choice([640, 1024, 1920]),
        "height": r.choice([480, 768, 1080]),
        "format": "PNG",
    },
    "IMAGE_RESIZE": lambda a, r: {
        "path": a.get("path"),
        "width": a.get("width"),
        "height": a.get("height"),
        "output_path": f"{a.get('path')}.resized.png",
        "note": "No image was written. Synthetic server.",
    },
    "IMAGE_CROP": lambda a, r: {
        "path": a.get("path"),
        "box": {
            "x": a.get("x"),
            "y": a.get("y"),
            "width": a.get("width"),
            "height": a.get("height"),
        },
        "output_path": f"{a.get('path')}.cropped.png",
        "note": "No image was written. Synthetic server.",
    },
    "IMAGE_FORMAT_CONVERT": lambda a, r: {
        "path": a.get("path"),
        "format": a.get("format"),
        "output_path": f"{a.get('path')}.{str(a.get('format', 'png')).lower()}",
        "note": "No image was written. Synthetic server.",
    },
    "IMAGE_DESCRIBE": lambda a, r: {
        "path": a.get("path"),
        "description": "Synthetic description: an image containing shapes and text.",
        "tags": ["synthetic", "image", "demo"],
    },
    # --- utility (real, pure) ---
    "DATE_DIFFERENCE": lambda a, r: _date_difference(a),
    "DATE_ADD": lambda a, r: _date_add(a),
    "UUID_GENERATE": lambda a, r: {
        # uuid5 over a fixed namespace: deterministic, unlike uuid4.
        "uuid": str(uuid.uuid5(UUID_NAMESPACE, "utility_mcp:generate_uuid")),
        "version": 5,
        "note": "Deterministic synthetic UUID.",
    },
    "HASH_TEXT": lambda a, r: _hash_text(a),
    "BASE64_ENCODE": lambda a, r: {
        "data": a.get("data"),
        "encoded": base64.b64encode(str(a.get("data", "")).encode()).decode(),
    },
    "BASE64_DECODE": lambda a, r: _base64_decode(a),
}


def _calculator(a: dict[str, Any]) -> dict[str, Any]:
    expression = str(a.get("expression", ""))
    try:
        value = safe_eval(expression)
    except Exception as exc:  # noqa: BLE001
        return {"expression": expression, "error": f"could not evaluate: {exc}"}
    return {"expression": expression, "result": value}


def _describe(a: dict[str, Any]) -> dict[str, Any]:
    values = _numbers(a.get("values"))
    if not values:
        return {"error": "no numeric values supplied"}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _percentage_change(a: dict[str, Any]) -> dict[str, Any]:
    try:
        old = float(a.get("old_value"))
        new = float(a.get("new_value"))
    except (TypeError, ValueError):
        return {"error": "old_value and new_value must be numbers"}
    if old == 0:
        return {"old_value": old, "new_value": new, "error": "old_value is zero"}
    change = (new - old) / abs(old) * 100
    return {
        "old_value": old,
        "new_value": new,
        "percentage_change": round(change, 6),
        "direction": "increase" if change >= 0 else "decrease",
    }


def _compound_interest(a: dict[str, Any]) -> dict[str, Any]:
    try:
        principal = float(a.get("principal"))
        rate = float(a.get("annual_rate"))
        years = float(a.get("years"))
    except (TypeError, ValueError):
        return {"error": "principal, annual_rate and years must be numbers"}
    n = int(a.get("compounds_per_year") or 1)
    # Accept both 8 and 0.08 for "8%".
    rate_fraction = rate / 100 if rate > 1 else rate
    amount = principal * (1 + rate_fraction / n) ** (n * years)
    return {
        "principal": principal,
        "annual_rate": rate,
        "years": years,
        "compounds_per_year": n,
        "final_amount": round(amount, 2),
        "interest_earned": round(amount - principal, 2),
    }


def _solve_linear(a: dict[str, Any]) -> dict[str, Any]:
    """Solve ``ax + b = cx + d`` by evaluating each side at x=0 and x=1."""
    equation = str(a.get("equation", ""))
    if "=" not in equation:
        return {"equation": equation, "error": "equation must contain '='"}
    left, right = equation.split("=", 1)

    def side(expr: str, x: float) -> float:
        cleaned = expr.replace("^", "**")
        # Insert explicit multiplication for the common "3x" form.
        out, prev = "", ""
        for ch in cleaned:
            if ch == "x" and prev.isdigit():
                out += "*"
            out += ch
            prev = ch
        return safe_eval(out.replace("x", f"({x})"))

    try:
        l0, l1 = side(left, 0.0), side(left, 1.0)
        r0, r1 = side(right, 0.0), side(right, 1.0)
    except Exception as exc:  # noqa: BLE001
        return {"equation": equation, "error": f"could not parse: {exc}"}

    slope = (l1 - l0) - (r1 - r0)
    if slope == 0:
        return {"equation": equation, "solution": None, "note": "no unique solution"}
    return {"equation": equation, "variable": "x", "solution": (r0 - l0) / slope}


def _json_parse(a: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"valid": True, "parsed": json.loads(str(a.get("json", "")))}
    except json.JSONDecodeError as exc:
        return {"valid": False, "error": str(exc)}


def _json_query(a: dict[str, Any]) -> dict[str, Any]:
    data = a.get("data")
    path = str(a.get("path", ""))
    cursor: Any = data
    for part in [p for p in path.replace("$", "").replace("[", ".").replace("]", "").split(".") if p]:
        if isinstance(cursor, dict):
            cursor = cursor.get(part)
        elif isinstance(cursor, list) and part.isdigit():
            index = int(part)
            cursor = cursor[index] if index < len(cursor) else None
        else:
            cursor = None
        if cursor is None:
            break
    return {"path": path, "value": cursor, "found": cursor is not None}


def _json_validate(a: dict[str, Any]) -> dict[str, Any]:
    """Shallow required/type check -- enough to be useful, no jsonschema dep."""
    data = a.get("data") or {}
    schema = a.get("schema") or {}
    errors: list[str] = []
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"missing required field: {field}")
    for field, spec in (schema.get("properties") or {}).items():
        if field not in data:
            continue
        expected = spec.get("type")
        python_type = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }.get(expected)
        if python_type and not isinstance(data[field], python_type):
            errors.append(f"{field}: expected {expected}")
    return {"valid": not errors, "errors": errors}


def _json_compare(a: dict[str, Any]) -> dict[str, Any]:
    left = a.get("left") or {}
    right = a.get("right") or {}
    left_keys, right_keys = set(left), set(right)
    return {
        "equal": left == right,
        "only_in_left": sorted(left_keys - right_keys),
        "only_in_right": sorted(right_keys - left_keys),
        "changed": sorted(k for k in left_keys & right_keys if left[k] != right[k]),
    }


def _date_difference(a: dict[str, Any]) -> dict[str, Any]:
    start = _parse_datetime(a.get("start"))
    end = _parse_datetime(a.get("end"))
    if start is None or end is None:
        return {"error": "start and end must be parseable dates"}
    delta = end - start
    return {
        "start": a.get("start"),
        "end": a.get("end"),
        "days": delta.days,
        "seconds": int(delta.total_seconds()),
    }


def _date_add(a: dict[str, Any]) -> dict[str, Any]:
    base = _parse_datetime(a.get("date"))
    if base is None:
        return {"error": "date must be parseable"}
    try:
        amount = int(a.get("amount"))
    except (TypeError, ValueError):
        return {"error": "amount must be an integer"}
    unit = str(a.get("unit", "days")).lower().rstrip("s")
    if unit in ("day", "week", "hour", "minute", "second"):
        delta = dt.timedelta(**{f"{unit}s": amount * (7 if unit == "week" else 1)})
        if unit == "week":
            delta = dt.timedelta(days=amount * 7)
        result = base + delta
    elif unit == "month":
        month_index = base.month - 1 + amount
        year = base.year + month_index // 12
        month = month_index % 12 + 1
        day = min(base.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        result = base.replace(year=year, month=month, day=day)
    elif unit == "year":
        result = base.replace(year=base.year + amount)
    else:
        return {"error": f"unsupported unit {unit!r}"}
    return {"date": a.get("date"), "amount": amount, "unit": a.get("unit"), "result": result.isoformat()}


def _hash_text(a: dict[str, Any]) -> dict[str, Any]:
    algorithm = str(a.get("algorithm", "sha256")).lower().replace("-", "")
    if algorithm not in hashlib.algorithms_available:
        return {"error": f"unsupported algorithm {a.get('algorithm')!r}"}
    digest = hashlib.new(algorithm, str(a.get("text", "")).encode()).hexdigest()
    return {"algorithm": algorithm, "hash": digest}


def _base64_decode(a: dict[str, Any]) -> dict[str, Any]:
    try:
        decoded = base64.b64decode(str(a.get("data", "")), validate=True)
    except (binascii.Error, ValueError) as exc:
        return {"error": f"invalid base64: {exc}"}
    return {"decoded": decoded.decode("utf-8", errors="replace")}


def _fallback(server_id: str, tool_id: str, arguments: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    return {
        "message": f"Synthetic response from {server_id}.{tool_id}",
        "echoed_arguments": arguments,
        "reference": _fake_id("syn", rng),
    }


def synthetic_result(
    server_id: str,
    tool_id: str,
    arguments: dict[str, Any],
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Deterministic fake execution envelope for one downstream tool call."""
    arguments = {k: v for k, v in (arguments or {}).items() if v is not None}
    rng = _rng(server_id, tool_id, arguments)

    handler = HANDLERS.get(operation_id or "")
    result = handler(arguments, rng) if handler else _fallback(server_id, tool_id, arguments, rng)

    return {
        "success": True,
        "synthetic": True,
        "server": server_id,
        "tool": tool_id,
        "operation": operation_id,
        "arguments": arguments,
        "result": result,
    }
