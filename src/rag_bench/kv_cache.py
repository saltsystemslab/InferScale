"""Per-chunk disk cache for pre-encoded RAG KV chunks, plus the CPU store.

Unlike the LoCoMo cache (kv/chunk_cache.py, one file per sample) this cache
stores ONE FILE PER CHUNK so the precompute pass is resumable per chunk.
The disk cache is the precompute artifact only: at answer time the full
corpus KV is loaded once into host RAM (CpuChunkStore, the cpu store
backend) and there is no answer-time disk I/O. The full-corpus KV does not
fit GPU HBM at MultiHop-RAG scale (roughly 180 GiB for Llama-3.1-8B in
bf16), but it fits the reference host's RAM.

Cache identity lives in the directory key (model, dtype, chunk_size,
context_window, max_position, corpus fingerprint) and is re-validated against
every payload's meta plus the exact chunk token ids on load, so a stale cache
can never silently inject wrong content.

torch is imported inside save/load functions so the module's pure key and
validation helpers stay importable and testable without torch.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from locomo_jasper_bench.cache_identity import atomic_write_json, safe_path_part
from locomo_jasper_bench.kv.types import EncodedChunk
from locomo_jasper_bench.runtime_paths import default_cache_root

from .data_types import RagChunk

RAG_CHUNK_CACHE_VERSION = 1

SCAFFOLD_SLOTS = ("header", "empty_passages", "footer")

_CHUNK_FIELDS = (
    "token_ids",
    "context_turn_ids",
    "context_prefix_tokens",
    "raw_context_prefix_tokens",
    "context_prefix_truncated_tokens",
)


def rag_chunk_cache_dir(
    *,
    model: str,
    dtype: str,
    chunk_size: int,
    context_window: int,
    max_position: int,
    corpus_fingerprint: str,
    cache_root: str | Path | None = None,
) -> Path:
    root = Path(cache_root) if cache_root is not None else default_cache_root() / "rag-kv-chunks"
    model_slug = safe_path_part(model.replace("/", "__"))
    return (
        root
        / f"v{RAG_CHUNK_CACHE_VERSION}"
        / model_slug
        / f"{dtype}-c{chunk_size}-w{context_window}-p{max_position}"
        / corpus_fingerprint
    )


def cache_meta_base(
    *,
    dataset: str,
    model: str,
    dtype: str,
    chunk_size: int,
    context_window: int,
    max_position: int,
    corpus_fingerprint: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "model": model,
        "dtype": dtype,
        "chunk_size": int(chunk_size),
        "context_window": int(context_window),
        "max_position": int(max_position),
        "corpus_fingerprint": corpus_fingerprint,
    }


def chunk_meta(meta_base: Mapping[str, Any], chunk: RagChunk) -> dict[str, Any]:
    return {**meta_base, "chunk_id": chunk.chunk_id, "token_count": chunk.token_count}


def tables_meta(meta_base: Mapping[str, Any], *, block_size: int) -> dict[str, Any]:
    return {**meta_base, "block_size": int(block_size)}


def chunk_file_path(cache_dir: Path, chunk_id: str) -> Path:
    return cache_dir / "chunks" / f"{safe_path_part(chunk_id)}.pt"


def tables_scaffold_path(cache_dir: Path, block_size: int) -> Path:
    return cache_dir / f"tables-scaffold-b{block_size}.pt"


def manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "manifest.json"


def chunk_payload_is_valid(
    payload: Any,
    *,
    expected_meta: Mapping[str, Any],
    expected_token_ids: Sequence[int],
) -> bool:
    """Pure structural validation shared by load and tests."""
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != RAG_CHUNK_CACHE_VERSION:
        return False
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return False
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            return False
    chunk = payload.get("chunk")
    if not isinstance(chunk, dict):
        return False
    if list(chunk.get("token_ids") or []) != list(expected_token_ids):
        return False
    if not isinstance(chunk.get("kv_by_layer"), dict) or not chunk["kv_by_layer"]:
        return False
    return True


def tables_payload_is_valid(payload: Any, *, expected_meta: Mapping[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != RAG_CHUNK_CACHE_VERSION:
        return False
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return False
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            return False
    if payload.get("cos_table") is None or payload.get("sin_table") is None:
        return False
    scaffold_chunks = payload.get("scaffold_chunks")
    if not isinstance(scaffold_chunks, dict):
        return False
    return all(scaffold_chunks.get(slot) is not None for slot in SCAFFOLD_SLOTS)


def save_chunk(path: Path, *, meta: Mapping[str, Any], chunk: EncodedChunk) -> None:
    import torch

    payload = {
        "version": RAG_CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "chunk": _chunk_to_payload(chunk),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so an interrupted precompute never leaves a truncated
    # file that later loads as a corrupt payload.
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_chunk(
    path: Path,
    *,
    expected_meta: Mapping[str, Any],
    expected_token_ids: Sequence[int],
    device: str = "cpu",
) -> EncodedChunk | None:
    """Return the cached chunk, or None on any miss or mismatch."""
    if not path.exists():
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning("Ignoring unreadable RAG KV chunk cache {}: {}", path, exc)
        return None
    if not chunk_payload_is_valid(
        payload, expected_meta=expected_meta, expected_token_ids=expected_token_ids
    ):
        logger.warning("Ignoring stale RAG KV chunk cache {} (metadata mismatch)", path)
        return None
    return _chunk_from_payload(payload["chunk"], device=device)


@dataclass(slots=True)
class CachedRagScaffold:
    cos_table: Any
    sin_table: Any
    scaffold_chunks: dict[str, EncodedChunk]


def save_tables_and_scaffold(
    path: Path,
    *,
    meta: Mapping[str, Any],
    cos_table: Any,
    sin_table: Any,
    scaffold_chunks: Mapping[str, EncodedChunk],
) -> None:
    import torch

    missing = [slot for slot in SCAFFOLD_SLOTS if scaffold_chunks.get(slot) is None]
    if missing:
        raise ValueError(f"Missing scaffold chunk slots: {missing}.")
    payload = {
        "version": RAG_CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "cos_table": cos_table.detach().to("cpu"),
        "sin_table": sin_table.detach().to("cpu"),
        "scaffold_chunks": {
            slot: _chunk_to_payload(scaffold_chunks[slot]) for slot in SCAFFOLD_SLOTS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    logger.info("Saved RAG KV tables and scaffold to {}", path)


def load_tables_and_scaffold(
    path: Path,
    *,
    expected_meta: Mapping[str, Any],
    device: str,
) -> CachedRagScaffold | None:
    if not path.exists():
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning("Ignoring unreadable RAG KV tables cache {}: {}", path, exc)
        return None
    if not tables_payload_is_valid(payload, expected_meta=expected_meta):
        logger.warning("Ignoring stale RAG KV tables cache {} (metadata mismatch)", path)
        return None
    scaffold_chunks = {
        slot: _chunk_from_payload(payload["scaffold_chunks"][slot], device=device)
        for slot in SCAFFOLD_SLOTS
    }
    return CachedRagScaffold(
        cos_table=payload["cos_table"].to(device),
        sin_table=payload["sin_table"].to(device),
        scaffold_chunks=scaffold_chunks,
    )


def rag_scaffold_chunks_match(
    scaffold_chunks: Mapping[str, EncodedChunk | None],
    scaffold: Any,
) -> bool:
    """Cached scaffold chunk token ids must equal the live tokenizer's scaffold."""
    expected = {
        "header": list(scaffold.header_token_ids),
        "empty_passages": list(scaffold.empty_memory_token_ids),
        "footer": list(scaffold.footer_token_ids),
    }
    for slot, expected_token_ids in expected.items():
        chunk = scaffold_chunks.get(slot)
        if chunk is None or list(chunk.token_ids) != expected_token_ids:
            return False
    return True


def write_manifest(
    cache_dir: Path,
    *,
    meta: Mapping[str, Any],
    chunks: Sequence[RagChunk],
) -> None:
    atomic_write_json(
        manifest_path(cache_dir),
        {
            "version": RAG_CHUNK_CACHE_VERSION,
            "meta": dict(meta),
            "chunk_count": len(chunks),
            "total_tokens": sum(chunk.token_count for chunk in chunks),
            "chunks": {chunk.chunk_id: chunk.token_count for chunk in chunks},
        },
        indent=2,
    )


def missing_chunk_files(cache_dir: Path, chunk_ids: Sequence[str]) -> list[str]:
    return [
        chunk_id
        for chunk_id in chunk_ids
        if not chunk_file_path(cache_dir, chunk_id).exists()
    ]


def encoded_chunk_nbytes(chunk: EncodedChunk) -> int:
    total = 0
    for tensor in chunk.kv_by_layer.values():
        nbytes = getattr(tensor, "nbytes", None)
        if nbytes is not None:
            total += int(nbytes)
    return total


class CpuChunkStore:
    """The cpu store backend: the full corpus chunk KV resident in host RAM.

    Loaded once, upfront, from the per-chunk precompute cache into pageable
    CPU tensors; compose_chunks copies the selected chunks to the GPU per
    query. There is no eviction and no answer-time disk I/O, so per-query
    latencies are uniform from the first query.
    """

    _LOG_EVERY = 200

    def __init__(
        self,
        cache_dir: Path,
        *,
        meta_base: Mapping[str, Any],
        chunks: Sequence[RagChunk],
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._meta_base = dict(meta_base)
        self._entries: dict[str, EncodedChunk] = {}
        self.resident_bytes = 0
        self.chunk_count = len(chunks)

        missing = [
            chunk.chunk_id
            for chunk in chunks
            if not chunk_file_path(self._cache_dir, chunk.chunk_id).exists()
        ]
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(chunks)} RAG KV chunk files are missing under "
                f"{self._cache_dir} (first missing: {missing[0]}). Run rag-jasper-bench "
                "--precompute-kv-only with the same --model, --chunk-size, and "
                "--context-window first."
            )
        expected_file_bytes = sum(
            chunk_file_path(self._cache_dir, chunk.chunk_id).stat().st_size
            for chunk in chunks
        )
        _warn_if_low_host_memory(expected_file_bytes)
        logger.info(
            "Loading {} RAG KV chunks ({:.1f} GiB on disk) into host RAM from {}",
            len(chunks),
            expected_file_bytes / 1024**3,
            self._cache_dir,
        )
        started = time.perf_counter()
        for index, chunk in enumerate(chunks, start=1):
            path = chunk_file_path(self._cache_dir, chunk.chunk_id)
            encoded = load_chunk(
                path,
                expected_meta=chunk_meta(self._meta_base, chunk),
                expected_token_ids=chunk.token_ids,
                device="cpu",
            )
            if encoded is None:
                raise RuntimeError(
                    f"RAG KV chunk cache is stale for chunk {chunk.chunk_id} at {path}. "
                    "Run rag-jasper-bench --precompute-kv-only with the same --model, "
                    "--chunk-size, and --context-window first."
                )
            self._entries[chunk.chunk_id] = encoded
            self.resident_bytes += encoded_chunk_nbytes(encoded)
            if index % self._LOG_EVERY == 0 or index == len(chunks):
                logger.info(
                    "Loaded {}/{} KV chunks into host RAM ({:.1f} GiB)",
                    index,
                    len(chunks),
                    self.resident_bytes / 1024**3,
                )
        self.load_time_ms = (time.perf_counter() - started) * 1000

    def fetch(self, chunk_ids: Sequence[str]) -> list[EncodedChunk]:
        fetched: list[EncodedChunk] = []
        for chunk_id in chunk_ids:
            encoded = self._entries.get(chunk_id)
            if encoded is None:
                raise RuntimeError(f"Retrieved chunk id {chunk_id} is not part of the corpus.")
            fetched.append(encoded)
        return fetched

    def stats(self) -> dict[str, Any]:
        return {
            "kv_store_backend": "cpu",
            "kv_store_chunk_count": self.chunk_count,
            "kv_store_resident_bytes": self.resident_bytes,
            "kv_store_load_time_ms": self.load_time_ms,
        }


def _warn_if_low_host_memory(expected_bytes: int) -> None:
    total = _host_memory_bytes()
    if total is not None and expected_bytes > 0.9 * total:
        logger.warning(
            "RAG KV chunks need about {:.1f} GiB of host RAM but this host reports "
            "{:.1f} GiB total; the cpu store may not fit.",
            expected_bytes / 1024**3,
            total / 1024**3,
        )


def _host_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return int(page_size) * int(page_count)


def _chunk_to_payload(chunk: EncodedChunk) -> dict[str, Any]:
    payload: dict[str, Any] = {field: getattr(chunk, field) for field in _CHUNK_FIELDS}
    payload["context_turn_ids"] = list(payload["context_turn_ids"])
    payload["kv_by_layer"] = {
        layer_name: tensor.detach().to("cpu")
        for layer_name, tensor in chunk.kv_by_layer.items()
    }
    return payload


def _chunk_from_payload(payload: dict[str, Any], *, device: str) -> EncodedChunk:
    return EncodedChunk(
        token_ids=list(payload["token_ids"]),
        kv_by_layer={
            layer_name: tensor.to(device)
            for layer_name, tensor in payload["kv_by_layer"].items()
        },
        context_turn_ids=tuple(payload["context_turn_ids"]),
        context_prefix_tokens=int(payload["context_prefix_tokens"]),
        raw_context_prefix_tokens=int(payload["raw_context_prefix_tokens"]),
        context_prefix_truncated_tokens=int(payload["context_prefix_truncated_tokens"]),
    )
