from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROVIDER_DEFAULT_ENDPOINT = "<provider-default>"


def normalize_endpoint(value: str | None) -> str:
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
    normalized = normalize_endpoint(endpoint)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"endpoint-{digest}"


def payload_header_matches(payload: Any, *, version: int, expected_meta: Mapping[str, Any]) -> bool:
    if not isinstance(payload, dict) or payload.get("version") != version:
        return False
    meta = payload.get("meta")
    return isinstance(meta, dict) and all(
        meta.get(key) == value for key, value in expected_meta.items()
    )


def safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "default"
