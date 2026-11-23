from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from inferscale.v1.types import SearchHit

from .types import RetrievedFact


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_value(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def memory_fact_from_hit(hit: SearchHit) -> RetrievedFact:
    memory_id = str(hit.id).strip()
    if not memory_id:
        raise RuntimeError("Mem0 returned a fact with an empty memory id.")
    raw_text = hit.payload.get("memory")
    text = str(raw_text).strip() if raw_text is not None else ""
    if not text:
        raise RuntimeError(f"Mem0 fact {memory_id} has no canonical payload['memory'] text.")

    session_index = _required_index(
        hit.payload, ("source_session_index", "session_index"), memory_id=memory_id
    )
    session_id = payload_value(hit.payload, ("source_session_id", "session_id"))
    turn_id = str(payload_value(hit.payload, ("source_turn_id", "turn_id")) or "").strip()
    raw_turn_index = payload_value(hit.payload, ("source_turn_index", "turn_index"))
    if raw_turn_index is None and turn_id:
        raw_turn_index = _trailing_turn_index(turn_id)
    turn_index = _index_value(raw_turn_index, name="source_turn_index", memory_id=memory_id)
    return RetrievedFact(
        memory_id=memory_id,
        text=text,
        text_hash=text_hash(text),
        created_at=str(hit.payload.get("created_at") or "").strip(),
        source_session_index=session_index,
        source_session_id=str(session_id or f"session_{session_index}").strip(),
        source_turn_index=turn_index,
        source_turn_id=turn_id,
    )


def unique_memory_facts(hits: Sequence[SearchHit]) -> list[RetrievedFact]:
    facts: list[RetrievedFact] = []
    fact_by_id: dict[str, RetrievedFact] = {}
    for hit in hits:
        fact = memory_fact_from_hit(hit)
        existing = fact_by_id.get(fact.memory_id)
        if existing is None:
            fact_by_id[fact.memory_id] = fact
            facts.append(fact)
        elif existing != fact:
            raise RuntimeError(f"Conflicting Mem0 fact payloads share memory id {fact.memory_id}.")
    return facts


def reverse_ranked_memory_facts(hits: Sequence[SearchHit]) -> list[RetrievedFact]:
    return list(reversed(unique_memory_facts(hits)))


def _trailing_turn_index(turn_id: str) -> str:
    return turn_id.rsplit(":", 1)[-1]


def _required_index(payload: Mapping[str, Any], keys: Sequence[str], *, memory_id: str) -> int:
    return _index_value(payload_value(payload, keys), name=keys[0], memory_id=memory_id)


def _index_value(raw_index: Any, *, name: str, memory_id: str) -> int:
    try:
        index = int(raw_index)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Mem0 fact {memory_id} has no valid {name}.") from exc
    if index < 0:
        raise RuntimeError(f"Mem0 fact {memory_id} has negative {name}={index}.")
    return index
