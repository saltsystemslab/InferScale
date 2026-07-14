from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def normalize_endpoint(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return "<provider-default>"
    parsed = urlsplit(text)
    if not parsed.scheme or not parsed.netloc:
        return text.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.query,
            "",
        )
    )


def endpoint_cache_key(endpoint: str | None) -> str:
    normalized = normalize_endpoint(endpoint)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"endpoint-{digest}"


def safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "default"


def atomic_write_json(path: Path, payload: Any, *, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            json.dump(
                payload,
                fh,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=indent,
                separators=(",", ":") if indent is None else None,
            )
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
