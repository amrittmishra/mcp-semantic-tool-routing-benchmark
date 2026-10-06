"""Single place where environment configuration is read.

Importing this module also imports :mod:`vertex_env`, which populates
``os.environ`` from the repo-root ``.env`` without overriding anything the
process was already started with. That means Docker ``env_file`` / ``environment``
values and ``source activate.sh`` both still win.
"""

from __future__ import annotations

import os
import pathlib

import vertex_env  # noqa: F401  (side effect: loads .env into os.environ)

ROOT = pathlib.Path(vertex_env.ROOT)


def _path(env_var: str, default: pathlib.Path) -> pathlib.Path:
    raw = os.getenv(env_var)
    return pathlib.Path(raw) if raw else default


def _flag(env_var: str, default: bool) -> bool:
    raw = os.getenv(env_var)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# --- paths -----------------------------------------------------------------
# Defaults are repo-root relative so local runs work with no extra config;
# Docker overrides them with /app/... absolutes.
REGISTRY_PATH = _path("REGISTRY_PATH", ROOT / "data" / "tools.json")
# Provider profiles live apart from the tool registry on purpose: the startup
# check hashes tools.json, so storing a reputation update there would invalidate
# the FAISS index on every observation.
PROVIDERS_PATH = _path("PROVIDERS_PATH", ROOT / "data" / "providers.json")
# Branch 2 registry: same operations, several real servers implementing each.
# Branch 1 keeps using REGISTRY_PATH untouched so published results reproduce.
QOS_REGISTRY_PATH = _path("QOS_REGISTRY_PATH", ROOT / "data" / "tools_qos.json")
CACHE_DIR = _path("CACHE_DIR", ROOT / "data" / "cache")
LOG_DIR = _path("LOG_DIR", ROOT / "logs")

INDEX_PATH = CACHE_DIR / "tools.faiss"
METADATA_PATH = CACHE_DIR / "tools_metadata.json"
MANIFEST_PATH = CACHE_DIR / "embedding_manifest.json"
RUN_LOG_PATH = LOG_DIR / "router_runs.jsonl"

# --- models ----------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "768"))
GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# --- router ----------------------------------------------------------------
ROUTER_TOP_K = int(os.getenv("ROUTER_TOP_K", "5"))
ROUTER_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTER_CONFIDENCE_THRESHOLD", "0.60"))
ROUTER_AMBIGUITY_MARGIN = float(os.getenv("ROUTER_AMBIGUITY_MARGIN", "0.03"))
ROUTER_DEBUG = _flag("ROUTER_DEBUG", False)

AUTO_BUILD_INDEX = _flag("AUTO_BUILD_INDEX", False)

# --- quality of service (Branch 2) ----------------------------------------
# Off by default: Branch 1 results must stay reproducible without touching env.
QOS_ENABLED = _flag("QOS_ENABLED", False)
# softmax reaches incentive fidelity 1.000; argmax caps at 0.584 because
# providers below rank one get no traffic and therefore no improvement gradient.
QOS_POLICY = os.getenv("QOS_POLICY", "softmax")
# 0.05 suits the shipped registry, whose provider profiles are already measured.
# A market learning profiles from scratch needs 0.10-0.25, or optimistic priors;
# see the tau discussion in orchestrator/qos.py.
QOS_TEMPERATURE = float(os.getenv("QOS_TEMPERATURE", "0.05"))
QOS_TASK_CLASS = os.getenv("QOS_TASK_CLASS", "default")
# Sampling makes repeated identical requests select different providers, which
# is correct for a marketplace but confusing in a demonstration. When true the
# highest-probability provider is taken instead.
QOS_DETERMINISTIC = _flag("QOS_DETERMINISTIC", True)
QOS_LEARNING_RATE = float(os.getenv("QOS_LEARNING_RATE", "0.05"))
QOS_SEED = int(os.getenv("QOS_SEED", "11"))

# --- downstream ------------------------------------------------------------
DOWNSTREAM_TIMEOUT_SECONDS = float(os.getenv("DOWNSTREAM_TIMEOUT_SECONDS", "30"))

# --- orchestrator http -----------------------------------------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
