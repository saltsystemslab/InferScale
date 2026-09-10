"""Disk cache for pre-encoded per-fact KV chunks, one file per LoCoMo sample.

Chunks are deterministic given (encoder model, dtype, max_position,
context_window, block_size, sample content, fact catalog content), so runs
reuse them instead of reloading the HF encoder and re-encoding every fact.
Retrieval and composition stay live; only the offline precompute is cached.
The precompute-kv stage prewarms the cache at setup time; run-time misses
still encode and save inline.

torch is imported only through the library payload helpers, so the pure key
helpers stay importable and testable on machines without torch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from loguru import logger

from inferscale.v1.kv.prompt import ScaffoldTokens
from inferscale.v1.kv.serialization import (
    chunk_from_payload,
    chunk_to_payload,
    load_torch_payload,
    save_torch_payload,
)
from inferscale.v1.types import KVChunk

from benchmarks.common.config import load_runtime_config

from .data import ConversationSample
from .mem0.fact_catalog import sample_fingerprint
from .types import MemoryFact

CHUNK_CACHE_VERSION = 1

# Scaffold slots stored in every payload. The composer consumes header,
# empty_memory, and footer; memory_list_header is always None (the scaffold
# renders retrieved facts without a list heading) and stays in the payload
# so the on-disk format is stable.
SCAFFOLD_SLOTS = ("header", "memory_list_header", "empty_memory", "footer")


@dataclass(slots=True)
class CachedSampleEncode:
    """One sample's cached encode, tensors already on the target device."""

    fact_chunks: dict[str, KVChunk]
    scaffold_chunks: dict[str, KVChunk | None]
    cos_table: Any
    sin_table: Any


def facts_fingerprint(facts: Sequence[MemoryFact]) -> str:
    """Order-sensitive content hash of the unique kv facts feeding the encode."""
    hasher = hashlib.sha256()
    for fact in facts:
        for part in (fact.memory_id, fact.text_hash, fact.source_turn_id):
            hasher.update(part.encode("utf-8"))
            hasher.update(b"\x00")
    return hasher.hexdigest()[:20]


def chunk_cache_dir(
    *,
    model: str,
    dtype: str,
    context_window: int,
    max_position: int,
    block_size: int,
    cache_root: str | Path | None = None,
) -> Path:
    root = Path(cache_root) if cache_root is not None else load_runtime_config().layout.cache_root
    model_slug = model.replace("/", "__")
    return (
        root
        / "kv-chunks"
        / f"v{CHUNK_CACHE_VERSION}"
        / model_slug
        / f"{dtype}-w{context_window}-p{max_position}-b{block_size}"
    )


def cache_path_for(
    *,
    model: str,
    dtype: str,
    context_window: int,
    max_position: int,
    block_size: int,
    sample: ConversationSample,
    facts: Sequence[MemoryFact],
    cache_root: str | Path | None = None,
) -> Path:
    directory = chunk_cache_dir(
        model=model,
        dtype=dtype,
        context_window=context_window,
        max_position=max_position,
        block_size=block_size,
        cache_root=cache_root,
    )
    return directory / f"{sample.sample_id}-{sample_fingerprint(sample)}-{facts_fingerprint(facts)}.pt"


def scaffold_chunks_match(scaffold_chunks: Mapping[str, KVChunk | None], scaffold: ScaffoldTokens) -> bool:
    """Cached scaffold chunk token ids must equal the live tokenizer's scaffold."""
    expected = {
        "header": list(scaffold.header_token_ids),
        "memory_list_header": [],
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
    """Pure structural validation shared by load and tests."""
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != CHUNK_CACHE_VERSION:
        return False
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return False
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            return False
    fact_chunks = payload.get("fact_chunks")
    scaffold_chunks = payload.get("scaffold_chunks")
    if not isinstance(fact_chunks, dict) or not isinstance(scaffold_chunks, dict):
        return False
    if set(fact_chunks) != set(expected_fact_ids):
        return False
    if scaffold_chunks.get("header") is None or scaffold_chunks.get("footer") is None:
        return False
    if payload.get("cos_table") is None or payload.get("sin_table") is None:
        return False
    return True


def save_sample_chunks(
    path: Path,
    *,
    meta: Mapping[str, Any],
    fact_chunks: Mapping[str, KVChunk],
    scaffold_chunks: Mapping[str, KVChunk | None],
    cos_table: Any,
    sin_table: Any,
) -> None:
    payload = {
        "version": CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "cos_table": cos_table.detach().to("cpu"),
        "sin_table": sin_table.detach().to("cpu"),
        "fact_chunks": {
            fact_id: chunk_to_payload(chunk) for fact_id, chunk in fact_chunks.items()
        },
        "scaffold_chunks": {
            slot: chunk_to_payload(chunk) if chunk is not None else None
            for slot in SCAFFOLD_SLOTS
            for chunk in [scaffold_chunks.get(slot)]
        },
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
    """Return the cached encode, or None on any miss.

    Fact chunks land on `device`; pass "cpu" when they are headed into the
    pinned-host chunk store so the disk payload never touches the GPU.
    Scaffold chunks and the RoPE tables land on `scaffold_device` (default:
    `device`) since composition consumes them directly on the GPU.
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
    scaffold_chunks = {
        slot: (
            chunk_from_payload(slot, payload["scaffold_chunks"][slot], device=scaffold_target)
            if payload["scaffold_chunks"].get(slot) is not None
            else None
        )
        for slot in SCAFFOLD_SLOTS
    }
    return CachedSampleEncode(
        fact_chunks=fact_chunks,
        scaffold_chunks=scaffold_chunks,
        cos_table=payload["cos_table"].to(scaffold_target),
        sin_table=payload["sin_table"].to(scaffold_target),
    )


def cache_meta(
    *,
    model: str,
    dtype: str,
    context_window: int,
    max_position: int,
    block_size: int,
    sample: ConversationSample,
    facts: Sequence[MemoryFact],
) -> dict[str, Any]:
    """The key metadata stored in and validated against every payload."""
    return {
        "model": model,
        "dtype": dtype,
        "context_window": int(context_window),
        "max_position": int(max_position),
        "block_size": int(block_size),
        "sample_id": sample.sample_id,
        "sample_fingerprint": sample_fingerprint(sample),
        "facts_fingerprint": facts_fingerprint(facts),
    }
