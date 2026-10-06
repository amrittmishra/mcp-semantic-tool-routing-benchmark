"""Load this folder's Vertex AI config, regardless of the process CWD.

Any Python program in this folder can just do::

    import vertex_env
    client = vertex_env.client()

or, if it builds its own client::

    import vertex_env  # side effect: os.environ is populated

Importing is idempotent and never overrides variables already set in the
environment, so `source activate.sh` (or real CI secrets) still wins.
"""

import functools
import os
import pathlib

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent

load_dotenv(ROOT / ".env", override=False)

# The Google SDKs resolve this against the process CWD, not the repo, so a
# relative value in .env silently breaks programs launched from elsewhere.
_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if _creds and not os.path.isabs(_creds):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(ROOT / _creds)

USE_VERTEX = os.getenv("USE_VERTEX_AI", "true").lower() in ("true", "1", "yes")
PROJECT = os.getenv("GCP_PROJECT_ID", "raisegate")
LOCATION = os.getenv("GCP_LOCATION", "global")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")


@functools.lru_cache(maxsize=1)
def client():
    """Return a google-genai Client pointed at Vertex, matching tracker-backend.

    Cached: the Client owns an httpx connection pool that is closed when the
    object is garbage collected, so a throwaway instance can die mid-request.
    """
    from google import genai

    return genai.Client(vertexai=USE_VERTEX, project=PROJECT, location=LOCATION)


EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_MAX_DIMS = 3072


def embed(texts, dims=None, task_type="RETRIEVAL_DOCUMENT", normalize=True):
    """Embed one string or a list of strings; returns a list of vectors.

    `dims` may be anything from 1 to 3072 (Matryoshka truncation of the same
    underlying vector). Only the full 3072-dim output is unit-length -- every
    truncated size comes back un-normalized, so `normalize` rescales by default.
    Pass normalize=False if your vector store normalizes on ingest.

    Use task_type="RETRIEVAL_QUERY" for lookups and "RETRIEVAL_DOCUMENT" for
    stored text; mixing them up quietly degrades retrieval quality.
    """
    from google.genai import types

    if isinstance(texts, str):
        texts = [texts]
    if dims is not None and not 1 <= dims <= EMBED_MAX_DIMS:
        raise ValueError(f"dims must be 1..{EMBED_MAX_DIMS}, got {dims}")

    config = types.EmbedContentConfig(task_type=task_type)
    if dims is not None:
        config.output_dimensionality = dims

    resp = client().models.embed_content(
        model=EMBED_MODEL, contents=texts, config=config
    )
    vectors = [e.values for e in resp.embeddings]

    if normalize:
        vectors = [
            [x / n for x in v] if (n := sum(x * x for x in v) ** 0.5) else v
            for v in vectors
        ]
    return vectors
