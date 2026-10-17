"""Content-addressed disk cache in front of an embedding model.

Every embedding the benchmarks need is computed once by the preembed stage and
written here, so timed runs read vectors from disk and never call the remote
embedding service. In read mode a missing or corrupt entry is an error rather
than a silent recomputation.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from benchmarks.common.atomic_files import atomic_write_npy
from benchmarks.common.cache_keys import endpoint_cache_key, normalize_endpoint, safe_path_part

CacheMode = Literal["read", "write"]
DEFAULT_PURPOSE = "default"


class CachedEmbeddingMissingError(RuntimeError):
    """Read mode found no usable cached vector for an input."""


class CachedEmbedder:
    """Wrap an embedder so repeated inputs are served from ``cache_dir``.

    The wrapped embedder is either a Mem0 embedder (``embed(text, memory_action)``
    and ``embed_batch(texts, memory_action)``) or the library's ``Embedder``
    (``embed_documents`` and ``embed_query``). Cache keys include the purpose,
    ``"add"`` for stored text and ``"search"`` for queries, so one cache serves
    both the Mem0 path and the library's retriever.
    """

    DOCUMENT_PURPOSE = "add"
    QUERY_PURPOSE = "search"

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
        self._calls = _wrapped_calls(wrapped)
        self.endpoint = normalize_endpoint(endpoint)
        self.cache_dir = Path(cache_dir) / safe_path_part(model) / endpoint_cache_key(self.endpoint)
        self.model = model
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self.embed_time_ms = 0.0
        if self.mode == "write":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        # Mem0 reads provider attributes such as ``config`` from its embedder.
        return getattr(self._wrapped, name)

    def embed_documents(self, texts: Any) -> list[list[float]]:
        return self.embed_batch(texts, self.DOCUMENT_PURPOSE)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_array(text, self.QUERY_PURPOSE)

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self.embed_array(text, memory_action).tolist()

    def embed_array(self, text: str, memory_action: str | None = None) -> np.ndarray:
        started = time.perf_counter()
        try:
            return self._embed_one(_require_text(text), memory_action)
        finally:
            self.embed_time_ms += (time.perf_counter() - started) * 1000

    def embed_batch(self, texts: Any, memory_action: str = DOCUMENT_PURPOSE) -> list[list[float]]:
        started = time.perf_counter()
        try:
            text_list = _text_list(texts)
            if not text_list:
                return []
            plan = self._plan_batch(text_list, memory_action)
            if plan.missing_texts:
                self._fetch_missing(plan, memory_action)
            return [vector.tolist() for vector in plan.vectors]
        finally:
            self.embed_time_ms += (time.perf_counter() - started) * 1000

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.mode,
            "endpoint": self.endpoint,
            "cache_dir": str(self.cache_dir),
            "hits": self.hits,
            "misses": self.misses,
        }

    def _embed_one(self, text: str, memory_action: str | None) -> np.ndarray:
        path = self._cache_path(text, memory_action)
        cached = self._load_cached(path)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        self._require_write_mode(memory_action)
        array = np.asarray(self._calls.embed_one(text, memory_action), dtype=np.float32)
        atomic_write_npy(path, array)
        return array

    def _plan_batch(self, texts: list[str], memory_action: str) -> _BatchPlan:
        """Resolve cache hits and group the misses by unique input, in first-seen order."""
        plan = _BatchPlan(vectors=[None] * len(texts))
        loaded: dict[Path, np.ndarray] = {}
        for index, text in enumerate(texts):
            path = self._cache_path(text, memory_action)
            if path in loaded:
                self.hits += 1
                plan.vectors[index] = loaded[path]
                continue
            if path in plan.positions_by_path:
                self.misses += 1
                plan.positions_by_path[path].append(index)
                continue
            cached = self._load_cached(path)
            if cached is not None:
                self.hits += 1
                loaded[path] = cached
                plan.vectors[index] = cached
                continue
            self.misses += 1
            self._require_write_mode(memory_action)
            plan.missing_texts.append(text)
            plan.missing_paths.append(path)
            plan.positions_by_path[path] = [index]
        return plan

    def _fetch_missing(self, plan: _BatchPlan, memory_action: str) -> None:
        raw_vectors = self._calls.embed_many(plan.missing_texts, memory_action)
        if len(raw_vectors) != len(plan.missing_texts):
            raise ValueError(
                f"Wrapped embed_batch() returned {len(raw_vectors)} embeddings for "
                f"{len(plan.missing_texts)} inputs."
            )
        for raw_vector, path in zip(raw_vectors, plan.missing_paths):
            array = np.asarray(raw_vector, dtype=np.float32)
            atomic_write_npy(path, array)
            for index in plan.positions_by_path[path]:
                plan.vectors[index] = array

    def _load_cached(self, path: Path) -> np.ndarray | None:
        """The cached vector, or None when absent; corrupt files are replaced in write mode."""
        if not path.exists():
            return None
        try:
            return np.load(path).astype(np.float32, copy=False)
        except Exception as exc:
            if self.mode == "read":
                raise CachedEmbeddingMissingError(
                    f"Cached embedding at {path} could not be loaded. "
                    "Delete the corrupt file and rerun the preembed stage with the same "
                    "dataset, embedding model, and embedding cache dir."
                ) from exc
            path.unlink(missing_ok=True)
            return None

    def _require_write_mode(self, memory_action: str | None) -> None:
        if self.mode == "read":
            raise CachedEmbeddingMissingError(
                f"Missing cached embedding for model={self.model!r} "
                f"purpose={_purpose(memory_action)!r}. Run the preembed stage with the same "
                "dataset, embedding model, and embedding cache dir before running the benchmark."
            )

    def _cache_path(self, text: str, memory_action: str | None) -> Path:
        key = hashlib.sha256()
        for part in (self.model, self.endpoint, _purpose(memory_action), text):
            key.update(part.encode("utf-8"))
            key.update(b"\0")
        return self.cache_dir / f"{key.hexdigest()}.npy"


@dataclass(slots=True)
class _BatchPlan:
    vectors: list[np.ndarray | None]
    missing_texts: list[str] = field(default_factory=list)
    missing_paths: list[Path] = field(default_factory=list)
    positions_by_path: dict[Path, list[int]] = field(default_factory=dict)


class _Mem0EmbedderCalls:
    """Call shape of a Mem0 embedder: the memory action is passed positionally."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def embed_one(self, text: str, memory_action: str | None) -> Any:
        return self._wrapped.embed(text, memory_action)

    def embed_many(self, texts: list[str], memory_action: str) -> list[Any]:
        return list(self._wrapped.embed_batch(texts, memory_action))


class _LibraryEmbedderCalls:
    """Call shape of the library's ``Embedder`` protocol, which has no memory action."""

    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def embed_one(self, text: str, memory_action: str | None) -> Any:
        return self._wrapped.embed_documents([text])[0]

    def embed_many(self, texts: list[str], memory_action: str) -> list[Any]:
        return list(self._wrapped.embed_documents(texts))


def _wrapped_calls(wrapped: Any) -> _Mem0EmbedderCalls | _LibraryEmbedderCalls:
    # The two embedder families have no common base class, so the call shape is
    # chosen once here, by the method the wrapped object provides.
    if hasattr(wrapped, "embed_documents"):
        return _LibraryEmbedderCalls(wrapped)
    return _Mem0EmbedderCalls(wrapped)


def _purpose(memory_action: str | None) -> str:
    return DEFAULT_PURPOSE if memory_action is None else str(memory_action)


def _require_text(text: Any) -> str:
    if not isinstance(text, str):
        raise TypeError("The embedding cache serves text inputs only.")
    return text


def _text_list(texts: Any) -> list[str]:
    if isinstance(texts, str | bytes):
        raise TypeError("embed_batch() requires an iterable of texts, not a single string.")
    try:
        return [_require_text(text) for text in texts]
    except TypeError as exc:
        raise TypeError("embed_batch() requires an iterable of texts.") from exc
