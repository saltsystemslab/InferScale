"""Normalized identity parts shared by every on-disk cache.

Caches are keyed by the embedding or LLM endpoint and model, so two runs that
differ only in URL spelling or credentials share entries while runs against a
different service never do.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROVIDER_DEFAULT_ENDPOINT = "<provider-default>"


def normalize_endpoint(value: str | None) -> str:
    """Lowercase scheme and host, drop fragments and trailing slashes; empty means the default."""
    text = str(value or "").strip()
    if not text:
        return PROVIDER_DEFAULT_ENDPOINT
    parsed = urlsplit(text)
    if not parsed.scheme or not parsed.netloc:
        return text.rstrip("/")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, "")
    )


def endpoint_cache_key(endpoint: str | None) -> str:
    """A short directory name derived from the normalized endpoint."""
    normalized = normalize_endpoint(endpoint)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"endpoint-{digest}"


def payload_header_matches(payload: Any, *, version: int, expected_meta: Mapping[str, Any]) -> bool:
    """True when a cached payload carries ``version`` and every expected identity value."""
    if not isinstance(payload, dict) or payload.get("version") != version:
        return False
    meta = payload.get("meta")
    return isinstance(meta, dict) and all(
        meta.get(key) == value for key, value in expected_meta.items()
    )


def safe_path_part(value: str) -> str:
    """Replace every character that is unsafe in a directory name with an underscore."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "default"
