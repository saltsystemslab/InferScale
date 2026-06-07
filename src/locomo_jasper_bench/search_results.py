from __future__ import annotations

from typing import Any

from .vector_types import SearchHit

_MIRRORED_METADATA_KEYS = ("user_id", "sample_id", "turn_id", "session_id", "turn_index", "speaker", "timestamp", "role")


def mem0_results_to_search_hits(results: Any) -> list[SearchHit]:
    items = _mem0_result_items(results)
    hits: list[SearchHit] = []
    for rank, item in enumerate(items, start=1):
        payload = _mem0_item_payload(item)
        score = _float_value(_item_get(item, "score"), 0.0)
        hits.append(
            SearchHit(
                id=str(_item_get(item, "id") or rank),
                payload=payload,
                score=score,
                distance=_float_value(_item_get(item, "distance"), score),
                rank=rank,
            )
        )
    return hits


def _mem0_result_items(results: Any) -> list[Any]:
    if isinstance(results, dict):
        value = results.get("results", [])
        return value if isinstance(value, list) else []
    if isinstance(results, list):
        return results
    return []


def _mem0_item_payload(item: Any) -> dict[str, Any]:
    raw_payload = _item_get(item, "payload")
    if isinstance(raw_payload, dict):
        payload = dict(raw_payload)
    else:
        payload = {}

    memory = _item_get(item, "memory") or payload.get("memory") or payload.get("data") or payload.get("text") or ""
    payload.setdefault("memory", memory)
    payload.setdefault("data", memory)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    item_metadata = _item_get(item, "metadata")
    if isinstance(item_metadata, dict):
        metadata.update(item_metadata)
    if isinstance(item, dict):
        for key, value in item.items():
            if key not in {"id", "memory", "score", "distance", "payload", "metadata"}:
                payload.setdefault(key, value)
                metadata.setdefault(key, value)
    payload["metadata"] = metadata
    return _normalize_memory_payload(payload)


def _item_get(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _normalize_memory_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(payload or {})
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        metadata = dict(metadata)
    else:
        metadata = {}

    for key in _MIRRORED_METADATA_KEYS:
        top_value = normalized.get(key)
        metadata_value = metadata.get(key)
        if top_value is None and metadata_value is not None:
            normalized[key] = metadata_value
        elif metadata_value is None and top_value is not None:
            metadata[key] = top_value

    normalized["metadata"] = metadata
    return normalized


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
