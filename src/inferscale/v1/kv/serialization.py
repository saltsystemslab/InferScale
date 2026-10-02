"""Plain payload form of a KVChunk for disk caches.

Payloads contain only dicts, lists, tuples, and tensors so they round-trip
through ``torch.load(weights_only=True)``. The key names are part of the
on-disk cache format and never change.

torch is imported inside the functions so the key helpers stay importable
and testable on machines without torch.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..types import KVChunk

logger = logging.getLogger(__name__)

TOKEN_IDS_KEY = "token_ids"
CONTEXT_IDS_KEY = "context_turn_ids"
CONTEXT_PREFIX_TOKENS_KEY = "context_prefix_tokens"
RAW_CONTEXT_PREFIX_TOKENS_KEY = "raw_context_prefix_tokens"
CONTEXT_PREFIX_TRUNCATED_TOKENS_KEY = "context_prefix_truncated_tokens"
KV_BY_LAYER_KEY = "kv_by_layer"


def chunk_to_payload(chunk: KVChunk) -> dict[str, Any]:
    return {
        TOKEN_IDS_KEY: list(chunk.token_ids),
        CONTEXT_IDS_KEY: list(chunk.context_ids),
        CONTEXT_PREFIX_TOKENS_KEY: int(chunk.context_prefix_tokens),
        RAW_CONTEXT_PREFIX_TOKENS_KEY: int(chunk.raw_context_prefix_tokens),
        CONTEXT_PREFIX_TRUNCATED_TOKENS_KEY: int(chunk.context_prefix_truncated_tokens),
        KV_BY_LAYER_KEY: {
            layer_name: tensor.detach().to("cpu")
            for layer_name, tensor in chunk.kv_by_layer.items()
        },
    }


def chunk_from_payload(chunk_id: str, payload: Mapping[str, Any], *, device: str) -> KVChunk:
    return KVChunk(
        chunk_id=chunk_id,
        token_ids=list(payload[TOKEN_IDS_KEY]),
        kv_by_layer={
            layer_name: tensor.to(device)
            for layer_name, tensor in payload[KV_BY_LAYER_KEY].items()
        },
        context_ids=tuple(payload[CONTEXT_IDS_KEY]),
        context_prefix_tokens=int(payload[CONTEXT_PREFIX_TOKENS_KEY]),
        raw_context_prefix_tokens=int(payload[RAW_CONTEXT_PREFIX_TOKENS_KEY]),
        context_prefix_truncated_tokens=int(payload[CONTEXT_PREFIX_TRUNCATED_TOKENS_KEY]),
    )


def save_torch_payload(path: Path, payload: Any) -> None:
    """Write-then-rename so an interrupted save never leaves a truncated file."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_torch_payload(path: Path) -> Any | None:
    """Return the payload, or None when the file is missing or unreadable."""
    if not path.exists():
        return None
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning("Ignoring unreadable KV payload %s: %s", path, exc)
        return None
