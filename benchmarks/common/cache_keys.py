from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROVIDER_DEFAULT_ENDPOINT = "<provider-default>"
NO_FRAGMENT = ""
ENDPOINT_DIGEST_LENGTH = 16
UNSAFE_PATH_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")


def normalize_endpoint(value: str | None) -> str:
    endpoint = str(value or "").strip()
    if not endpoint:
        return PROVIDER_DEFAULT_ENDPOINT
    url = urlsplit(endpoint)
    if not url.scheme or not url.netloc:
        return endpoint.rstrip("/")
    return urlunsplit(
        (url.scheme.lower(), url.netloc.lower(), url.path.rstrip("/"), url.query, NO_FRAGMENT)
    )


def endpoint_cache_key(endpoint: str | None) -> str:
    normalized = normalize_endpoint(endpoint)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:ENDPOINT_DIGEST_LENGTH]
    return f"endpoint-{digest}"


def payload_header_matches(payload: Any, *, version: int, expected_meta: Mapping[str, Any]) -> bool:
    if not isinstance(payload, dict) or payload.get("version") != version:
        return False
    meta = payload.get("meta")
    return isinstance(meta, dict) and all(
        meta.get(name) == expected for name, expected in expected_meta.items()
    )


def safe_path_part(value: str) -> str:
    safe = UNSAFE_PATH_CHARACTERS.sub("_", value.strip())
    return safe or "default"
