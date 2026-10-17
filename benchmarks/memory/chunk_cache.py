"""Disk cache for pre-encoded per-fact KV chunks, one file per LoCoMo sample.

Chunks are deterministic given the encoder model, dtype, max position,
context window, block size, sample content, and fact catalog content, so runs
reuse them instead of reloading the HF encoder and re-encoding every fact.
Retrieval and composition stay live; only the offline precompute is cached.
The precompute-kv stage fills the cache ahead of time; run-time misses still
encode and save inline.

torch is imported only through the library payload helpers, so the key
helpers stay importable and testable on machines without torch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from benchmarks.common.cache_keys import payload_header_matches
from inferscale.v1.kv.prompt import ScaffoldTokens
from inferscale.v1.kv.serialization import (
    chunk_from_payload,
    chunk_to_payload,
    load_torch_payload,
    save_torch_payload,
)
from inferscale.v1.types import KVChunk

from .data import ConversationSample, sample_fingerprint
from .types import RetrievedFact

CHUNK_CACHE_VERSION = 2
SCAFFOLD_SLOTS = ("header", "empty_memory", "footer")


class KVEncodingSettings(Protocol):
    """The settings that determine a chunk's KV, as both benchmark configs expose them."""

    @property
    def model(self) -> str: ...

    @property
    def kv_dtype(self) -> str: ...

    @property
    def context_window(self) -> int: ...

    @property
    def kv_max_position(self) -> int: ...

    @property
    def kv_block_size(self) -> int: ...


@dataclass(slots=True, frozen=True)
class ChunkCacheKey:
    """Everything a cached encode depends on, naming its file and validating its payload."""

    model: str
    dtype: str
    context_window: int
    max_position: int
    block_size: int
    sample_id: str
    sample_fingerprint: str
    facts_fingerprint: str

    @classmethod
    def from_config(
        cls,
        config: KVEncodingSettings,
        sample: ConversationSample,
        facts: Sequence[RetrievedFact],
    ) -> ChunkCacheKey:
        return cls(
            model=config.model,
            dtype=config.kv_dtype,
            context_window=int(config.context_window),
            max_position=int(config.kv_max_position),
            block_size=int(config.kv_block_size),
            sample_id=sample.sample_id,
            sample_fingerprint=sample_fingerprint(sample),
            facts_fingerprint=facts_fingerprint(facts),
        )

    def directory(self, cache_root: Path) -> Path:
        """The directory shared by every sample encoded with these settings."""
        return (
            Path(cache_root)
            / "kv-chunks"
            / f"v{CHUNK_CACHE_VERSION}"
            / self.model.replace("/", "__")
            / f"{self.dtype}-w{self.context_window}-p{self.max_position}-b{self.block_size}"
        )

    def path(self, cache_root: Path) -> Path:
        name = f"{self.sample_id}-{self.sample_fingerprint}-{self.facts_fingerprint}.pt"
        return self.directory(cache_root) / name

    def meta(self) -> dict[str, Any]:
        """The key as stored in, and validated against, every payload."""
        return {
            "model": self.model,
            "dtype": self.dtype,
            "context_window": self.context_window,
            "max_position": self.max_position,
            "block_size": self.block_size,
            "sample_id": self.sample_id,
            "sample_fingerprint": self.sample_fingerprint,
            "facts_fingerprint": self.facts_fingerprint,
        }


@dataclass(slots=True)
class CachedSampleEncode:
    """One sample's cached encode, tensors already on the target device."""

    fact_chunks: dict[str, KVChunk]
    scaffold_chunks: dict[str, KVChunk | None]
    cos_table: Any
    sin_table: Any


def facts_fingerprint(facts: Sequence[RetrievedFact]) -> str:
    """Order-sensitive content hash of the unique kv facts feeding the encode."""
    hasher = hashlib.sha256()
    for fact in facts:
        for part in (fact.memory_id, fact.text_hash, fact.source_turn_id):
            hasher.update(part.encode("utf-8"))
            hasher.update(b"\x00")
    return hasher.hexdigest()[:20]


def scaffold_chunks_match(
    scaffold_chunks: Mapping[str, KVChunk | None], scaffold: ScaffoldTokens
) -> bool:
    """Cached scaffold chunk token ids must equal the live tokenizer's scaffold."""
    expected = {
        "header": list(scaffold.header_token_ids),
        "empty_memory": list(scaffold.empty_token_ids),
        "footer": list(scaffold.footer_token_ids),
    }
    for slot, expected_token_ids in expected.items():
        chunk = scaffold_chunks.get(slot)
        cached_token_ids = list(chunk.token_ids) if chunk is not None else []
        if cached_token_ids != expected_token_ids:
            return False
    return True


def payload_is_valid(
    payload: Any,
    *,
    expected_meta: Mapping[str, Any],
    expected_fact_ids: Sequence[str],
) -> bool:
    """Structural validation of a loaded payload, shared by load and tests."""
    if not payload_header_matches(
        payload, version=CHUNK_CACHE_VERSION, expected_meta=expected_meta
    ):
        return False
    fact_chunks = payload.get("fact_chunks")
    scaffold_chunks = payload.get("scaffold_chunks")
    if not isinstance(fact_chunks, dict) or not isinstance(scaffold_chunks, dict):
        return False
    if set(fact_chunks) != set(expected_fact_ids):
        return False
    if scaffold_chunks.get("header") is None or scaffold_chunks.get("footer") is None:
        return False
    return payload.get("cos_table") is not None and payload.get("sin_table") is not None


def save_sample_chunks(
    path: Path,
    *,
    meta: Mapping[str, Any],
    fact_chunks: Mapping[str, KVChunk],
    scaffold_chunks: Mapping[str, KVChunk | None],
    cos_table: Any,
    sin_table: Any,
) -> None:
    scaffold_payloads: dict[str, Any] = {}
    for slot in SCAFFOLD_SLOTS:
        chunk = scaffold_chunks.get(slot)
        scaffold_payloads[slot] = chunk_to_payload(chunk) if chunk is not None else None
    payload = {
        "version": CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "cos_table": cos_table.detach().to("cpu"),
        "sin_table": sin_table.detach().to("cpu"),
        "fact_chunks": {fact_id: chunk_to_payload(chunk) for fact_id, chunk in fact_chunks.items()},
        "scaffold_chunks": scaffold_payloads,
    }
    save_torch_payload(path, payload)
    logger.info(
        "Saved KV chunk cache {} ({} fact chunks, {:.1f} MB)",
        path,
        len(fact_chunks),
        path.stat().st_size / 1e6,
    )


def load_sample_chunks(
    path: Path,
    *,
    device: str,
    scaffold_device: str | None = None,
    expected_meta: Mapping[str, Any],
    expected_fact_ids: Sequence[str],
) -> CachedSampleEncode | None:
    """The cached encode, or None on any miss.

    Fact chunks land on ``device``; pass "cpu" when they are headed into the
    pinned-host chunk store so the disk payload never touches the GPU.
    Scaffold chunks and the RoPE tables land on ``scaffold_device`` (default
    ``device``) since composition consumes them directly on the GPU.
    """
    payload = load_torch_payload(path)
    if payload is None:
        return None
    if not payload_is_valid(
        payload, expected_meta=expected_meta, expected_fact_ids=expected_fact_ids
    ):
        logger.warning("Ignoring stale KV chunk cache {} (metadata mismatch)", path)
        return None
    scaffold_target = scaffold_device or device
    fact_chunks = {
        fact_id: chunk_from_payload(fact_id, chunk_payload, device=device)
        for fact_id, chunk_payload in payload["fact_chunks"].items()
    }
    scaffold_chunks: dict[str, KVChunk | None] = {}
    for slot in SCAFFOLD_SLOTS:
        chunk_payload = payload["scaffold_chunks"].get(slot)
        scaffold_chunks[slot] = (
            chunk_from_payload(slot, chunk_payload, device=scaffold_target)
            if chunk_payload is not None
            else None
        )
    return CachedSampleEncode(
        fact_chunks=fact_chunks,
        scaffold_chunks=scaffold_chunks,
        cos_table=payload["cos_table"].to(scaffold_target),
        sin_table=payload["sin_table"].to(scaffold_target),
    )
