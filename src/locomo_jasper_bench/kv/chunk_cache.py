"""Disk cache for pre-encoded per-fact KV chunks.

Chunks are deterministic given (encoder model, dtype, max_position,
context_window, block_size, sample content, fact catalog content), so both
benches can reuse them across runs instead of reloading the HF encoder and
re-encoding every fact. Retrieval and composition stay live; only the
offline precompute is cached. `scripts/precompute_kv_chunks.sh` prewarms
the cache at setup time; run-time misses still encode and save inline.

torch is imported inside the save/load functions so the module (and its
pure-python key helpers) stays importable and testable on machines without
torch, matching cpu_memory_store.py.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from loguru import logger

from ..data import ConversationSample
from ..retrieval.fact_catalog import sample_fingerprint
from ..runtime_paths import default_cache_root
from .types import EncodedChunk, MemoryFact

CHUNK_CACHE_VERSION = 1

# Scaffold slots stored in every payload; the accuracy composer consumes all
# four, the throughput condition only header/footer. memory_list_header may
# legitimately be None (scaffolds without a list heading).
SCAFFOLD_SLOTS = ("header", "memory_list_header", "empty_memory", "footer")

_CHUNK_FIELDS = (
    "token_ids",
    "context_turn_ids",
    "context_prefix_tokens",
    "raw_context_prefix_tokens",
    "context_prefix_truncated_tokens",
)


@dataclass(slots=True)
class CachedSampleEncode:
    """One sample's cached encode, tensors already on the target device."""

    fact_chunks: dict[str, EncodedChunk]
    scaffold_chunks: dict[str, EncodedChunk | None]
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
    root = Path(cache_root) if cache_root is not None else default_cache_root()
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


def scaffold_chunks_match(scaffold_chunks: Mapping[str, EncodedChunk | None], scaffold: Any) -> bool:
    """Cached scaffold chunk token ids must equal the live tokenizer's scaffold."""
    expected = {
        "header": list(scaffold.header_token_ids),
        "memory_list_header": list(scaffold.memory_list_header_token_ids or []),
        "empty_memory": list(scaffold.empty_memory_token_ids),
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
    fact_chunks: Mapping[str, EncodedChunk],
    scaffold_chunks: Mapping[str, EncodedChunk | None],
    cos_table: Any,
    sin_table: Any,
) -> None:
    import torch

    payload = {
        "version": CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "cos_table": cos_table.detach().to("cpu"),
        "sin_table": sin_table.detach().to("cpu"),
        "fact_chunks": {
            fact_id: _chunk_to_payload(chunk) for fact_id, chunk in fact_chunks.items()
        },
        "scaffold_chunks": {
            slot: _chunk_to_payload(scaffold_chunks.get(slot)) for slot in SCAFFOLD_SLOTS
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a crash mid-save can never leave a truncated file
    # that later loads as a corrupt payload.
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
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

    Fact chunks land on `device` - pass "cpu" when they are headed into the
    pinned-host chunk store so the disk payload never touches the GPU.
    Scaffold chunks and the RoPE tables land on `scaffold_device` (default:
    `device`) since composition consumes them directly on the GPU.
    """
    if not path.exists():
        return None
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning("Ignoring unreadable KV chunk cache {}: {}", path, exc)
        return None
    if not payload_is_valid(
        payload, expected_meta=expected_meta, expected_fact_ids=expected_fact_ids
    ):
        logger.warning("Ignoring stale KV chunk cache {} (metadata mismatch)", path)
        return None

    scaffold_target = scaffold_device or device
    fact_chunks = {
        fact_id: _chunk_from_payload(chunk_payload, device=device)
        for fact_id, chunk_payload in payload["fact_chunks"].items()
    }
    scaffold_chunks = {
        slot: _chunk_from_payload(payload["scaffold_chunks"].get(slot), device=scaffold_target)
        for slot in SCAFFOLD_SLOTS
    }
    return CachedSampleEncode(
        fact_chunks=fact_chunks,
        scaffold_chunks=scaffold_chunks,
        cos_table=payload["cos_table"].to(scaffold_target),
        sin_table=payload["sin_table"].to(scaffold_target),
    )


def _chunk_to_payload(chunk: EncodedChunk | None) -> dict[str, Any] | None:
    if chunk is None:
        return None
    payload: dict[str, Any] = {field: getattr(chunk, field) for field in _CHUNK_FIELDS}
    payload["kv_by_layer"] = {
        layer_name: tensor.detach().to("cpu")
        for layer_name, tensor in chunk.kv_by_layer.items()
    }
    return payload


def _chunk_from_payload(payload: dict[str, Any] | None, *, device: str) -> EncodedChunk | None:
    if payload is None:
        return None
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
