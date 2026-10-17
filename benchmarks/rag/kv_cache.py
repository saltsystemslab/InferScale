"""Per-chunk disk cache of pre-encoded RAG KV, keyed by everything an encode depends on.

One file per chunk keeps the precompute pass resumable per chunk; the RoPE
tables and the prompt scaffold chunks share one file per block size. The
cache is the precompute artifact only: at answer time the whole corpus KV is
loaded once into host RAM (cpu_chunk_store.py) and nothing is read from disk
per query.

Cache identity lives in the directory key (model, dtype, chunk size, context
window, max position, corpus fingerprint) and is checked again against every
payload's meta and the exact chunk token ids on load, so a stale cache can
never silently inject wrong content.

torch is imported only through the library payload helpers, so the key and
validation helpers stay importable and testable without torch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.cache_keys import payload_header_matches, safe_path_part
from benchmarks.rag.chunking import corpus_fingerprint
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import RagChunk, RagDocument
from inferscale.v1.kv.serialization import (
    chunk_from_payload,
    chunk_to_payload,
    load_torch_payload,
    save_torch_payload,
)
from inferscale.v1.types import KVChunk

RAG_CHUNK_CACHE_VERSION = 1
SCAFFOLD_SLOTS = ("header", "empty_passages", "footer")


@dataclass(slots=True, frozen=True)
class RagChunkCacheKey:
    """The settings a cached encode depends on; they name the directory and stamp every payload."""

    dataset: str
    model: str
    dtype: str
    chunk_size: int
    context_window: int
    max_position: int
    corpus_fingerprint: str

    @classmethod
    def from_config(cls, config: RagBenchConfig, docs: Sequence[RagDocument]) -> RagChunkCacheKey:
        return cls(
            dataset=config.dataset_name,
            model=config.model,
            dtype=config.kv_dtype,
            chunk_size=int(config.chunk_size),
            context_window=int(config.context_window),
            max_position=int(config.kv_max_position),
            corpus_fingerprint=corpus_fingerprint(docs),
        )

    def directory(self, cache_root: Path) -> Path:
        model_slug = safe_path_part(self.model.replace("/", "__"))
        settings = f"{self.dtype}-c{self.chunk_size}-w{self.context_window}-p{self.max_position}"
        return (
            Path(cache_root)
            / f"v{RAG_CHUNK_CACHE_VERSION}"
            / model_slug
            / settings
            / self.corpus_fingerprint
        )

    def meta(self) -> dict[str, Any]:
        """The identity stored in, and validated against, every payload."""
        return {
            "dataset": self.dataset,
            "model": self.model,
            "dtype": self.dtype,
            "chunk_size": self.chunk_size,
            "context_window": self.context_window,
            "max_position": self.max_position,
            "corpus_fingerprint": self.corpus_fingerprint,
        }

    def chunk_meta(self, chunk: RagChunk) -> dict[str, Any]:
        return {**self.meta(), "chunk_id": chunk.chunk_id, "token_count": chunk.token_count}

    def tables_meta(self, *, block_size: int) -> dict[str, Any]:
        return {**self.meta(), "block_size": int(block_size)}


@dataclass(slots=True, frozen=True)
class RagChunkCache:
    """One cache directory and the identity every payload in it must carry."""

    key: RagChunkCacheKey
    directory: Path

    @classmethod
    def for_config(cls, config: RagBenchConfig, docs: Sequence[RagDocument]) -> RagChunkCache:
        if config.kv_chunk_cache_root is None:
            raise RuntimeError(
                "rag.kv_chunk_cache_root is not set; load the run from JSON with a runtime."
            )
        key = RagChunkCacheKey.from_config(config, docs)
        return cls(key=key, directory=key.directory(config.kv_chunk_cache_root))

    def chunk_path(self, chunk_id: str) -> Path:
        return self.directory / "chunks" / f"{safe_path_part(chunk_id)}.pt"

    def tables_path(self, block_size: int) -> Path:
        return self.directory / f"tables-scaffold-b{block_size}.pt"

    @property
    def manifest_path(self) -> Path:
        return self.directory / "manifest.json"

    def missing_chunk_ids(self, chunk_ids: Iterable[str]) -> list[str]:
        return [chunk_id for chunk_id in chunk_ids if not self.chunk_path(chunk_id).exists()]

    def require_complete(self, chunks: Sequence[RagChunk]) -> None:
        """Fail with the precompute settings when any chunk file is absent."""
        missing = self.missing_chunk_ids(chunk.chunk_id for chunk in chunks)
        if not missing:
            return
        raise RuntimeError(
            f"{len(missing)} of {len(chunks)} RAG KV chunk files are missing under "
            f"{self.directory} (first missing: {missing[0]}). Run the precompute-kv stage for "
            f"dataset {self.key.dataset}, model {self.key.model}, chunk size "
            f"{self.key.chunk_size}, context window {self.key.context_window}."
        )

    def write_manifest(self, chunks: Sequence[RagChunk]) -> None:
        atomic_write_json(
            self.manifest_path,
            {
                "version": RAG_CHUNK_CACHE_VERSION,
                "meta": self.key.meta(),
                "chunk_count": len(chunks),
                "total_tokens": sum(chunk.token_count for chunk in chunks),
                "chunks": {chunk.chunk_id: chunk.token_count for chunk in chunks},
            },
            indent=2,
        )

    def total_bytes(self) -> int:
        """Bytes of the chunk and tables files on disk."""
        chunks_dir = self.directory / "chunks"
        chunk_files = chunks_dir.glob("*.pt") if chunks_dir.exists() else ()
        tables_files = self.directory.glob("tables-scaffold-*.pt")
        return sum(path.stat().st_size for path in (*chunk_files, *tables_files))


def chunk_payload_is_valid(
    payload: Any,
    *,
    expected_meta: Mapping[str, Any],
    expected_token_ids: Sequence[int],
) -> bool:
    """Structural validation of a chunk payload, shared by load and tests."""
    if not payload_header_matches(
        payload, version=RAG_CHUNK_CACHE_VERSION, expected_meta=expected_meta
    ):
        return False
    chunk = payload.get("chunk")
    if not isinstance(chunk, dict):
        return False
    if list(chunk.get("token_ids") or []) != list(expected_token_ids):
        return False
    return isinstance(chunk.get("kv_by_layer"), dict) and bool(chunk["kv_by_layer"])


def tables_payload_is_valid(payload: Any, *, expected_meta: Mapping[str, Any]) -> bool:
    if not payload_header_matches(
        payload, version=RAG_CHUNK_CACHE_VERSION, expected_meta=expected_meta
    ):
        return False
    if payload.get("cos_table") is None or payload.get("sin_table") is None:
        return False
    scaffold_chunks = payload.get("scaffold_chunks")
    if not isinstance(scaffold_chunks, dict):
        return False
    return all(scaffold_chunks.get(slot) is not None for slot in SCAFFOLD_SLOTS)


def save_chunk(path: Path, *, meta: Mapping[str, Any], chunk: KVChunk) -> None:
    payload = {
        "version": RAG_CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "chunk": chunk_to_payload(chunk),
    }
    save_torch_payload(path, payload)


def load_chunk(
    path: Path,
    *,
    expected_meta: Mapping[str, Any],
    expected_token_ids: Sequence[int],
    device: str = "cpu",
) -> KVChunk | None:
    """The cached chunk, or None on any miss or mismatch."""
    payload = load_torch_payload(path)
    if payload is None:
        return None
    if not chunk_payload_is_valid(
        payload, expected_meta=expected_meta, expected_token_ids=expected_token_ids
    ):
        logger.warning("Ignoring stale RAG KV chunk cache {} (metadata mismatch)", path)
        return None
    chunk_id = str(expected_meta.get("chunk_id") or path.stem)
    return chunk_from_payload(chunk_id, payload["chunk"], device=device)


@dataclass(slots=True)
class CachedRagScaffold:
    """The RoPE tables and the encoded prompt scaffold chunks of one cache."""

    cos_table: Any
    sin_table: Any
    scaffold_chunks: dict[str, KVChunk]


def save_tables_and_scaffold(
    path: Path,
    *,
    meta: Mapping[str, Any],
    cos_table: Any,
    sin_table: Any,
    scaffold_chunks: Mapping[str, KVChunk],
) -> None:
    missing = [slot for slot in SCAFFOLD_SLOTS if scaffold_chunks.get(slot) is None]
    if missing:
        raise ValueError(f"Missing scaffold chunk slots: {missing}.")
    payload = {
        "version": RAG_CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "cos_table": cos_table.detach().to("cpu"),
        "sin_table": sin_table.detach().to("cpu"),
        "scaffold_chunks": {
            slot: chunk_to_payload(scaffold_chunks[slot]) for slot in SCAFFOLD_SLOTS
        },
    }
    save_torch_payload(path, payload)
    logger.info("Saved RAG KV tables and scaffold to {}", path)


def load_tables_and_scaffold(
    path: Path,
    *,
    expected_meta: Mapping[str, Any],
    device: str,
) -> CachedRagScaffold | None:
    payload = load_torch_payload(path)
    if payload is None:
        return None
    if not tables_payload_is_valid(payload, expected_meta=expected_meta):
        logger.warning("Ignoring stale RAG KV tables cache {} (metadata mismatch)", path)
        return None
    scaffold_chunks = {
        slot: chunk_from_payload(slot, payload["scaffold_chunks"][slot], device=device)
        for slot in SCAFFOLD_SLOTS
    }
    return CachedRagScaffold(
        cos_table=payload["cos_table"].to(device),
        sin_table=payload["sin_table"].to(device),
        scaffold_chunks=scaffold_chunks,
    )


def rag_scaffold_chunks_match(
    scaffold_chunks: Mapping[str, KVChunk | None],
    scaffold: Any,
) -> bool:
    """Cached scaffold chunk token ids must equal the live tokenizer's scaffold."""
    expected = {
        "header": list(scaffold.header_token_ids),
        "empty_passages": list(scaffold.empty_token_ids),
        "footer": list(scaffold.footer_token_ids),
    }
    for slot, expected_token_ids in expected.items():
        chunk = scaffold_chunks.get(slot)
        if chunk is None or list(chunk.token_ids) != expected_token_ids:
            return False
    return True
