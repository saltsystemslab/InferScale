from __future__ import annotations

import importlib.metadata
import json
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, get_type_hints

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.cache_keys import endpoint_cache_key, normalize_endpoint, safe_path_part
from benchmarks.common.json_fields import typed_value
from benchmarks.common.vector_types import VECTOR_DISTANCE, SearchHit
from benchmarks.memory.data import (
    ConversationSample,
    Turn,
    locomo_timestamp,
    locomo_turn_role,
    sample_fingerprint,
)
from benchmarks.memory.protocol import (
    FACT_CATALOG_FORMAT_VERSION,
    MEMORY_EXTRACTION_MAX_FACTS,
    MEMORY_EXTRACTION_MAX_MODEL_LEN,
    MEMORY_EXTRACTION_MAX_TEXT_CHARS,
    MEMORY_EXTRACTION_MAX_TOKENS,
    MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
    MEMORY_INGESTION_PROTOCOL,
)

FACT_NAMESPACE = uuid.UUID("55bbc260-62d5-4a84-9724-6686fd41798e")
SOURCE_TURN_JOIN_ALIASES = ("session_id", "turn_index", "turn_id")
FACT_METADATA_KEYS = (
    "fact_id",
    "user_id",
    "sample_id",
    "created_at",
    "timestamp",
    "timestamp_epoch",
    "source_session_index",
    "source_session_id",
    "source_turn_index",
    "source_turn_id",
    "speaker",
    "source_role",
    "role",
    *SOURCE_TURN_JOIN_ALIASES,
)


@dataclass(slots=True, frozen=True)
class MemoryFact:
    id: str
    text: str
    created_at: str
    timestamp_epoch: int
    sample_id: str
    source_session_index: int
    source_session_id: str
    source_turn_index: int
    source_turn_id: str
    speaker: str
    role: str

    def metadata(self) -> dict[str, Any]:
        return {
            "fact_id": self.id,
            **source_fields(
                sample_id=self.sample_id,
                created_at=self.created_at,
                timestamp_epoch=self.timestamp_epoch,
                session_index=self.source_session_index,
                session_id=self.source_session_id,
                turn_index=self.source_turn_index,
                turn_id=self.source_turn_id,
                speaker=self.speaker,
                role=self.role,
            ),
        }

    def to_search_hit(self, rank: int) -> SearchHit:
        return SearchHit(
            id=self.id,
            payload=fact_payload(self, self.metadata()),
            score=0.0,
            distance=0.0,
            rank=rank,
        )


FACT_FIELD_TYPES = get_type_hints(MemoryFact)


def source_fields(
    *,
    sample_id: str,
    created_at: str,
    timestamp_epoch: int,
    session_index: int,
    session_id: str,
    turn_index: int,
    turn_id: str,
    speaker: str,
    role: str,
) -> dict[str, Any]:
    return {
        "user_id": sample_id,
        "sample_id": sample_id,
        "created_at": created_at,
        "timestamp": timestamp_epoch,
        "timestamp_epoch": timestamp_epoch,
        "source_session_index": session_index,
        "source_session_id": session_id,
        "source_turn_index": turn_index,
        "source_turn_id": turn_id,
        "speaker": speaker,
        "source_role": role,
        "role": role,
        "session_id": session_id,
        "turn_index": turn_index,
        "turn_id": turn_id,
    }


def source_metadata(sample: ConversationSample, turn: Turn) -> dict[str, Any]:
    timestamp_epoch, created_at = locomo_timestamp(turn.timestamp)
    return source_fields(
        sample_id=sample.sample_id,
        created_at=created_at,
        timestamp_epoch=timestamp_epoch,
        session_index=turn.session_index,
        session_id=turn.session_id,
        turn_index=turn.turn_index,
        turn_id=turn.id,
        speaker=turn.speaker,
        role=locomo_turn_role(sample, turn),
    )


def fact_payload(fact: MemoryFact, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory": fact.text,
        "text": fact.text,
        "data": fact.text,
        "created_at": fact.created_at,
        **metadata,
        "metadata": metadata,
    }


class FactCatalogStore:
    def __init__(
        self,
        root: str | Path,
        *,
        provider: str,
        model: str,
        endpoint: str | None,
        embedding_model: str,
        mem0_version: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.provider = provider.strip()
        self.model = model.strip()
        self.endpoint = normalize_endpoint(endpoint)
        self.embedding_model = embedding_model.strip()
        self.mem0_version = mem0_version or importlib.metadata.version("mem0ai")
        self.temperature = float(temperature)
        self.catalog_dir = (
            self.root
            / "fact-catalogs"
            / f"mem0-{safe_path_part(self.mem0_version)}"
            / safe_path_part(self.provider)
            / safe_path_part(self.model)
            / safe_path_part(MEMORY_EXTRACTION_RESPONSE_PROTOCOL)
            / f"context-{MEMORY_EXTRACTION_MAX_MODEL_LEN}"
            / f"tokens-{MEMORY_EXTRACTION_MAX_TOKENS}"
            / f"facts-{MEMORY_EXTRACTION_MAX_FACTS}"
            / f"text-{MEMORY_EXTRACTION_MAX_TEXT_CHARS}"
            / endpoint_cache_key(self.endpoint)
            / safe_path_part(self.embedding_model)
        )

    def path_for(self, sample: ConversationSample) -> Path:
        fingerprint = sample_fingerprint(sample)
        return self.catalog_dir / f"{safe_path_part(sample.sample_id)}-{fingerprint}.json"

    def write(self, sample: ConversationSample, facts: Iterable[MemoryFact]) -> Path:
        catalog_path = self.path_for(sample)
        catalog = {**self._identity(sample), "facts": [asdict(fact) for fact in facts]}
        atomic_write_json(catalog_path, catalog, indent=2)
        return catalog_path

    def load(self, sample: ConversationSample) -> tuple[MemoryFact, ...]:
        catalog_path = self.path_for(sample)
        catalog = _read_catalog(catalog_path)
        for key, expected in self._identity(sample).items():
            if catalog.get(key) != expected:
                raise RuntimeError(
                    f"Mem0 fact catalog identity mismatch for {key} at {catalog_path}: "
                    f"expected={expected!r} actual={catalog.get(key)!r}. "
                    "Rerun the preembed stage with the current JSON configuration."
                )
        return _facts_from_rows(catalog.get("facts"), catalog_path)

    def _identity(self, sample: ConversationSample) -> dict[str, Any]:
        return {
            "version": FACT_CATALOG_FORMAT_VERSION,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "mem0_version": self.mem0_version,
            "memory_llm_temperature": self.temperature,
            "memory_extraction_response_protocol": MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
            "memory_extraction_max_model_len": MEMORY_EXTRACTION_MAX_MODEL_LEN,
            "memory_extraction_max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
            "memory_extraction_max_facts": MEMORY_EXTRACTION_MAX_FACTS,
            "memory_extraction_max_text_chars": MEMORY_EXTRACTION_MAX_TEXT_CHARS,
            "vector_distance": VECTOR_DISTANCE,
            "ingestion_protocol": MEMORY_INGESTION_PROTOCOL,
            "embedding_model": self.embedding_model,
            "sample_id": sample.sample_id,
            "sample_fingerprint": sample_fingerprint(sample),
        }


def _read_catalog(catalog_path: Path) -> dict[str, Any]:
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing Mem0 fact catalog at {catalog_path}. Rerun the extract-facts stage "
            "with the same dataset, memory LLM model, endpoint, and cache dir."
        ) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Corrupt Mem0 fact catalog at {catalog_path}. "
            "Rerun the extract-facts stage to replace it."
        ) from exc
    if not isinstance(catalog, dict):
        raise RuntimeError(f"Corrupt Mem0 fact catalog at {catalog_path}: not a JSON object.")
    return catalog


def _facts_from_rows(rows: Any, catalog_path: Path) -> tuple[MemoryFact, ...]:
    if not isinstance(rows, list):
        raise RuntimeError(f"Mem0 fact catalog has no facts list: {catalog_path}")
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"Mem0 fact catalog contains a non-object fact: {catalog_path}")
    try:
        facts = tuple(_fact_from_row(row) for row in rows)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Mem0 fact catalog contains an invalid fact: {catalog_path}") from exc
    fact_ids = [fact.id for fact in facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise RuntimeError(f"Mem0 fact catalog contains duplicate fact ids: {catalog_path}")
    return facts


def _fact_from_row(row: dict[str, Any]) -> MemoryFact:
    fact = MemoryFact(**row)
    for name, expected in FACT_FIELD_TYPES.items():
        typed_value(getattr(fact, name), name, expected, "fact")
    return fact


def make_memory_fact(text: str, sample: ConversationSample, turn: Turn) -> MemoryFact:
    normalized_text = " ".join(str(text).split())
    if not normalized_text:
        raise ValueError("Mem0 fact text must not be empty.")
    timestamp_epoch, created_at = locomo_timestamp(turn.timestamp)
    identity = json.dumps(
        {"sample_id": sample.sample_id, "source_turn_id": turn.id, "text": normalized_text},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return MemoryFact(
        id=str(uuid.uuid5(FACT_NAMESPACE, identity)),
        text=normalized_text,
        created_at=created_at,
        timestamp_epoch=timestamp_epoch,
        sample_id=sample.sample_id,
        source_session_index=turn.session_index,
        source_session_id=turn.session_id,
        source_turn_index=turn.turn_index,
        source_turn_id=turn.id,
        speaker=turn.speaker,
        role=locomo_turn_role(sample, turn),
    )


def fact_catalog_hits(facts: Iterable[MemoryFact]) -> list[SearchHit]:
    return [fact.to_search_hit(rank) for rank, fact in enumerate(facts, start=1)]
