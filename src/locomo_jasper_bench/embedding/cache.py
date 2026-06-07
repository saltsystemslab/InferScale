from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np

CacheMode = Literal["read", "write"]


class CachedEmbeddingMissingError(RuntimeError):
    pass


class CachedEmbedder:
    """Disk-backed wrapper for Mem0 embedders."""

    def __init__(self, wrapped: Any, *, cache_dir: str | Path, model: str, mode: CacheMode = "write") -> None:
        if mode not in {"read", "write"}:
            raise ValueError(f"Unsupported embedding cache mode: {mode}")
        self._wrapped = wrapped
        self.cache_dir = Path(cache_dir) / _safe_path_part(model)
        self.model = model
        self.mode = mode
        self.hits = 0
        self.misses = 0
        if self.mode == "write":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def embed(self, text: Any, *args: Any, **kwargs: Any) -> list[float]:
        if not isinstance(text, str):
            if self.mode == "read":
                raise CachedEmbeddingMissingError(
                    "Benchmark embedding cache is read-only and cannot serve non-string embedding input. "
                    "Run with --no-embedding-cache only for debugging."
                )
            return self._wrapped.embed(text, *args, **kwargs)

        purpose = _embedding_purpose(args, kwargs)
        path = self._cache_path(text, purpose)
        if path.exists():
            try:
                self.hits += 1
                return np.load(path).astype(np.float32, copy=False).tolist()
            except Exception as exc:
                self.hits -= 1
                if self.mode == "read":
                    raise CachedEmbeddingMissingError(
                        f"Cached embedding at {path} could not be loaded. "
                        "Delete the corrupt file and rerun locomo-jasper-bench --preembed-only with the same "
                        "dataset, embedding model, and embedding cache dir."
                    ) from exc
                path.unlink(missing_ok=True)

        self.misses += 1
        if self.mode == "read":
            raise CachedEmbeddingMissingError(
                f"Missing cached embedding for model={self.model!r} purpose={purpose!r}. "
                "Run locomo-jasper-bench --preembed-only with the same dataset, embedding model, "
                "and embedding cache dir before running the benchmark."
            )
        vector = self._wrapped.embed(text, *args, **kwargs)
        array = np.asarray(vector, dtype=np.float32)
        self._write_array(path, array)
        return array.tolist()

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.mode,
            "cache_dir": str(self.cache_dir),
            "hits": self.hits,
            "misses": self.misses,
        }

    def _cache_path(self, text: str, purpose: str) -> Path:
        key = hashlib.sha256()
        key.update(self.model.encode("utf-8"))
        key.update(b"\0")
        key.update(purpose.encode("utf-8"))
        key.update(b"\0")
        key.update(text.encode("utf-8"))
        return self.cache_dir / f"{key.hexdigest()}.npy"

    def _write_array(self, path: Path, array: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp_path.open("wb") as fh:
            np.save(fh, array.astype(np.float32, copy=False))
        os.replace(tmp_path, path)


def _embedding_purpose(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    for key in ("purpose", "memory_action", "mode"):
        value = kwargs.get(key)
        if value is not None:
            return str(value)
    if args and isinstance(args[0], str):
        return args[0]
    return "default"


def _safe_path_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe or "default"
