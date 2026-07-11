from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..cache_identity import endpoint_cache_key, normalize_endpoint, safe_path_part

CacheMode = Literal["read", "write"]


class CachedEmbeddingMissingError(RuntimeError):
    pass


class CachedEmbedder:
    """Disk-backed wrapper for Mem0 embedders."""

    def __init__(
        self,
        wrapped: Any,
        *,
        cache_dir: str | Path,
        model: str,
        mode: CacheMode = "write",
        endpoint: str | None = None,
    ) -> None:
        if mode not in {"read", "write"}:
            raise ValueError(f"Unsupported embedding cache mode: {mode}")
        self._wrapped = wrapped
        self.endpoint = normalize_endpoint(endpoint)
        self.cache_dir = (
            Path(cache_dir)
            / safe_path_part(model)
            / endpoint_cache_key(self.endpoint)
        )
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

        return self.embed_array(text, *args, **kwargs).tolist()

    def embed_array(self, text: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        if not isinstance(text, str):
            if self.mode == "read":
                raise CachedEmbeddingMissingError(
                    "Benchmark embedding cache is read-only and cannot serve non-string embedding input. "
                    "Run with --no-embedding-cache only for debugging."
                )
            return np.asarray(self._wrapped.embed(text, *args, **kwargs), dtype=np.float32)

        purpose = _embedding_purpose(args, kwargs)
        path = self._cache_path(text, purpose)
        if path.exists():
            try:
                self.hits += 1
                return np.load(path).astype(np.float32, copy=False)
            except Exception as exc:
                self.hits -= 1
                if self.mode == "read":
                    self.misses += 1
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
        return array

    def embed_batch(self, texts: Any, *args: Any, **kwargs: Any) -> list[list[float]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("embed_batch() requires an iterable of embedding inputs, not a single string.")
        try:
            text_list = list(texts)
        except TypeError as exc:
            raise TypeError("embed_batch() requires an iterable of embedding inputs.") from exc
        if not text_list:
            return []

        purpose = _embedding_purpose(args, kwargs)
        vectors: list[np.ndarray | None] = [None] * len(text_list)
        loaded_by_path: dict[Path, np.ndarray] = {}
        missing_inputs: list[Any] = []
        missing_indices: list[list[int]] = []
        missing_paths: list[Path | None] = []
        missing_group_by_path: dict[Path, int] = {}

        for index, text in enumerate(text_list):
            if not isinstance(text, str):
                if self.mode == "read":
                    raise CachedEmbeddingMissingError(
                        "Benchmark embedding cache is read-only and cannot serve non-string embedding input. "
                        "Run with --no-embedding-cache only for debugging."
                    )
                self.misses += 1
                missing_inputs.append(text)
                missing_indices.append([index])
                missing_paths.append(None)
                continue

            path = self._cache_path(text, purpose)
            loaded = loaded_by_path.get(path)
            if loaded is not None:
                self.hits += 1
                vectors[index] = loaded
                continue

            missing_group = missing_group_by_path.get(path)
            if missing_group is not None:
                self.misses += 1
                missing_indices[missing_group].append(index)
                continue

            if path.exists():
                try:
                    loaded = np.load(path).astype(np.float32, copy=False)
                except Exception as exc:
                    self.misses += 1
                    if self.mode == "read":
                        raise CachedEmbeddingMissingError(
                            f"Cached embedding at {path} could not be loaded. "
                            "Delete the corrupt file and rerun locomo-jasper-bench --preembed-only with the same "
                            "dataset, embedding model, and embedding cache dir."
                        ) from exc
                    path.unlink(missing_ok=True)
                else:
                    self.hits += 1
                    loaded_by_path[path] = loaded
                    vectors[index] = loaded
                    continue
            else:
                self.misses += 1

            if self.mode == "read":
                raise CachedEmbeddingMissingError(
                    f"Missing cached embedding for model={self.model!r} purpose={purpose!r}. "
                    "Run locomo-jasper-bench --preembed-only with the same dataset, embedding model, "
                    "and embedding cache dir before running the benchmark."
                )
            missing_group_by_path[path] = len(missing_inputs)
            missing_inputs.append(text)
            missing_indices.append([index])
            missing_paths.append(path)

        if missing_inputs:
            wrapped_embed_batch = getattr(self._wrapped, "embed_batch", None)
            if callable(wrapped_embed_batch):
                raw_vectors = list(wrapped_embed_batch(missing_inputs, *args, **kwargs))
            else:
                raw_vectors = [self._wrapped.embed(text, *args, **kwargs) for text in missing_inputs]
            if len(raw_vectors) != len(missing_inputs):
                raise ValueError(
                    f"Wrapped embed_batch() returned {len(raw_vectors)} embeddings for "
                    f"{len(missing_inputs)} inputs."
                )

            for raw_vector, indices, path in zip(raw_vectors, missing_indices, missing_paths, strict=True):
                array = np.asarray(raw_vector, dtype=np.float32)
                if path is not None:
                    self._write_array(path, array)
                for index in indices:
                    vectors[index] = array

        result: list[list[float]] = []
        for vector in vectors:
            if vector is None:
                raise RuntimeError("Embedding cache failed to resolve every batch input.")
            result.append(vector.tolist())
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.mode,
            "endpoint": self.endpoint,
            "cache_dir": str(self.cache_dir),
            "hits": self.hits,
            "misses": self.misses,
        }

    def _cache_path(self, text: str, purpose: str) -> Path:
        key = hashlib.sha256()
        key.update(self.model.encode("utf-8"))
        key.update(b"\0")
        key.update(self.endpoint.encode("utf-8"))
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
