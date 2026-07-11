from __future__ import annotations

import hashlib
import importlib.metadata
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..cache_identity import atomic_write_json, endpoint_cache_key, normalize_endpoint, safe_path_part
from ..data import ConversationSample, Turn
from ..protocol import MEMORY_INGESTION_PROTOCOL
from ..vector_types import SearchHit, VECTOR_DISTANCE


_FACT_NAMESPACE = uuid.UUID("55bbc260-62d5-4a84-9724-6686fd41798e")
_CATALOG_VERSION = 4
_LOCOMO_TIMESTAMP_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
    "%Y-%m-%d",
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
            "user_id": self.sample_id,
            "sample_id": self.sample_id,
            "created_at": self.created_at,
            "timestamp": self.timestamp_epoch,
            "timestamp_epoch": self.timestamp_epoch,
            "source_session_index": self.source_session_index,
            "source_session_id": self.source_session_id,
            "source_turn_index": self.source_turn_index,
            "source_turn_id": self.source_turn_id,
            "speaker": self.speaker,
            "source_role": self.role,
            "role": self.role,
            # Aliases consumed by the KV/prefix source-turn join.
            "session_id": self.source_session_id,
            "turn_index": self.source_turn_index,
            "turn_id": self.source_turn_id,
        }

    def to_search_hit(self, rank: int) -> SearchHit:
        metadata = self.metadata()
        return SearchHit(
            id=self.id,
            payload={
                "memory": self.text,
                "text": self.text,
                "data": self.text,
                "created_at": self.created_at,
                **metadata,
                "metadata": metadata,
            },
            score=0.0,
            distance=0.0,
            rank=rank,
        )


class FactCatalogStore:
    def __init__(
        self,
        root: str | Path,
        *,
        provider: str,
        model: str,
        endpoint: str | None,
        embedding_model: str,
        embedding_endpoint: str | None,
        mem0_version: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.root = Path(root)
        self.provider = provider.strip()
        self.model = model.strip()
        self.endpoint = normalize_endpoint(endpoint)
        self.embedding_model = embedding_model.strip()
        self.embedding_endpoint = normalize_endpoint(embedding_endpoint)
        self.mem0_version = mem0_version or importlib.metadata.version("mem0ai")
        self.temperature = float(temperature)
        self.catalog_dir = (
            self.root
            / "fact-catalogs"
            / f"mem0-{safe_path_part(self.mem0_version)}"
            / safe_path_part(self.provider)
            / safe_path_part(self.model)
            / endpoint_cache_key(self.endpoint)
            / safe_path_part(self.embedding_model)
            / endpoint_cache_key(self.embedding_endpoint)
        )

    def path_for(self, sample: ConversationSample) -> Path:
        fingerprint = sample_fingerprint(sample)
        return self.catalog_dir / f"{safe_path_part(sample.sample_id)}-{fingerprint}.json"

    def write(
        self,
        sample: ConversationSample,
        facts: Iterable[MemoryFact],
    ) -> Path:
        path = self.path_for(sample)
        payload = {
            "version": _CATALOG_VERSION,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "mem0_version": self.mem0_version,
            "memory_llm_temperature": self.temperature,
            "vector_distance": VECTOR_DISTANCE,
            "ingestion_protocol": MEMORY_INGESTION_PROTOCOL,
            "embedding_model": self.embedding_model,
            "embedding_endpoint": self.embedding_endpoint,
            "sample_id": sample.sample_id,
            "sample_fingerprint": sample_fingerprint(sample),
            "facts": [asdict(fact) for fact in facts],
        }
        atomic_write_json(path, payload, indent=2)
        return path

    def load(self, sample: ConversationSample) -> tuple[MemoryFact, ...]:
        path = self.path_for(sample)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Missing Mem0 fact catalog at {path}. Rerun locomo-jasper-bench "
                "--preembed-only with the same dataset, memory LLM model, endpoint, and cache dir."
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Corrupt Mem0 fact catalog at {path}. Rerun locomo-jasper-bench "
                "--preembed-only to replace it."
            ) from exc

        expected = {
            "version": _CATALOG_VERSION,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "mem0_version": self.mem0_version,
            "memory_llm_temperature": self.temperature,
            "vector_distance": VECTOR_DISTANCE,
            "ingestion_protocol": MEMORY_INGESTION_PROTOCOL,
            "embedding_model": self.embedding_model,
            "embedding_endpoint": self.embedding_endpoint,
            "sample_id": sample.sample_id,
            "sample_fingerprint": sample_fingerprint(sample),
        }
        for key, expected_value in expected.items():
            if payload.get(key) != expected_value:
                raise RuntimeError(
                    f"Mem0 fact catalog identity mismatch for {key} at {path}: "
                    f"expected={expected_value!r} actual={payload.get(key)!r}. "
                    "Rerun --preembed-only with the current configuration."
                )
        rows = payload.get("facts")
        if not isinstance(rows, list):
            raise RuntimeError(f"Mem0 fact catalog has no facts list: {path}")
        try:
            facts = tuple(MemoryFact(**row) for row in rows if isinstance(row, dict))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Mem0 fact catalog contains an invalid fact: {path}") from exc
        if len(facts) != len(rows):
            raise RuntimeError(f"Mem0 fact catalog contains a non-object fact: {path}")
        ids = [fact.id for fact in facts]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Mem0 fact catalog contains duplicate fact ids: {path}")
        return facts


def make_memory_fact(text: str, sample: ConversationSample, turn: Turn) -> MemoryFact:
    normalized_text = " ".join(str(text).split())
    if not normalized_text:
        raise ValueError("Mem0 fact text must not be empty.")
    timestamp_epoch, created_at = locomo_timestamp(turn.timestamp)
    role = locomo_turn_role(sample, turn)
    identity = json.dumps(
        {
            "sample_id": sample.sample_id,
            "source_turn_id": turn.id,
            "text": normalized_text,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return MemoryFact(
        id=str(uuid.uuid5(_FACT_NAMESPACE, identity)),
        text=normalized_text,
        created_at=created_at,
        timestamp_epoch=timestamp_epoch,
        sample_id=sample.sample_id,
        source_session_index=turn.session_index,
        source_session_id=turn.session_id,
        source_turn_index=turn.turn_index,
        source_turn_id=turn.id,
        speaker=turn.speaker,
        role=role,
    )


def fact_catalog_hits(facts: Iterable[MemoryFact]) -> list[SearchHit]:
    return [fact.to_search_hit(rank) for rank, fact in enumerate(facts, start=1)]


def source_metadata(sample: ConversationSample, turn: Turn) -> dict[str, Any]:
    timestamp_epoch, created_at = locomo_timestamp(turn.timestamp)
    role = locomo_turn_role(sample, turn)
    return {
        "user_id": sample.sample_id,
        "sample_id": sample.sample_id,
        "created_at": created_at,
        "timestamp": timestamp_epoch,
        "timestamp_epoch": timestamp_epoch,
        "source_session_index": turn.session_index,
        "source_session_id": turn.session_id,
        "source_turn_index": turn.turn_index,
        "source_turn_id": turn.id,
        "speaker": turn.speaker,
        "source_role": role,
        "role": role,
        "session_id": turn.session_id,
        "turn_index": turn.turn_index,
        "turn_id": turn.id,
    }


def locomo_turn_role(sample: ConversationSample, turn: Turn) -> str:
    conversation = sample.raw.get("conversation", sample.raw)
    if not isinstance(conversation, dict):
        conversation = {}
    speaker_a = str(conversation.get("speaker_a") or "").strip()
    speaker_b = str(conversation.get("speaker_b") or "").strip()
    if speaker_a and turn.speaker == speaker_a:
        return "user"
    if speaker_b and turn.speaker == speaker_b:
        return "assistant"

    normalized = turn.speaker.strip().lower()
    if normalized in {"assistant", "ai", "bot"}:
        return "assistant"
    return "user"


def locomo_timestamp(value: str | None) -> tuple[int, str]:
    """Return a UTC epoch/ISO pair; generic records with no date use Unix epoch."""
    parsed: datetime | None = None
    text = str(value or "").strip()
    if text:
        iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(iso_text)
        except ValueError:
            for timestamp_format in _LOCOMO_TIMESTAMP_FORMATS:
                try:
                    parsed = datetime.strptime(text, timestamp_format)
                except ValueError:
                    continue
                break
    if parsed is None and text:
        raise ValueError(f"Unsupported LoCoMo timestamp: {text!r}.")
    if parsed is None:
        parsed = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elif parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp()), parsed.isoformat()


def sample_fingerprint(sample: ConversationSample) -> str:
    conversation = sample.raw.get("conversation", sample.raw)
    if not isinstance(conversation, dict):
        conversation = {}
    canonical = json.dumps(
        {
            "sample_id": sample.sample_id,
            "speaker_a": conversation.get("speaker_a"),
            "speaker_b": conversation.get("speaker_b"),
            "turns": [
                {
                    "id": turn.id,
                    "session_index": turn.session_index,
                    "turn_index": turn.turn_index,
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "timestamp": turn.timestamp,
                    "image_caption": turn.image_caption,
                }
                for turn in sample.turns
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
