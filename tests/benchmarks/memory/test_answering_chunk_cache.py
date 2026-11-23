from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

from benchmarks.memory import chunk_cache
from benchmarks.memory.chunk_cache import (
    CHUNK_CACHE_VERSION,
    ChunkCacheKey,
    facts_fingerprint,
    payload_is_valid,
    save_sample_chunks,
)
from benchmarks.memory.data import ConversationSample, sample_fingerprint
from benchmarks.memory.types import RetrievedFact
from inferscale.v1.types import KVChunk

META = {"model": "m", "sample_id": "conv-1"}


class Tensor:
    def __init__(self, name: str, device: str = "cuda:0", *, detached: bool = False) -> None:
        self.name = name
        self.device = device
        self.detached = detached

    def detach(self) -> Tensor:
        return Tensor(self.name, self.device, detached=True)

    def to(self, device: str) -> Tensor:
        return Tensor(self.name, device, detached=self.detached)


@pytest.fixture
def log() -> Iterator[list[tuple[str, str]]]:
    messages: list[tuple[str, str]] = []
    sink = logger.add(
        lambda message: messages.append((message.record["level"].name, message.record["message"])),
        level="INFO",
    )
    yield messages
    logger.remove(sink)


def _fact(memory_id: str, text_hash: str = "hash", turn_id: str = "turn-1") -> RetrievedFact:
    return RetrievedFact(
        memory_id=memory_id,
        text=f"text for {memory_id}",
        text_hash=text_hash,
        created_at="2023-05-08",
        source_session_index=1,
        source_session_id="session-1",
        source_turn_index=2,
        source_turn_id=turn_id,
    )


def _chunk(chunk_id: str, token_ids: list[int]) -> KVChunk:
    return KVChunk(
        chunk_id=chunk_id,
        token_ids=token_ids,
        kv_by_layer={"layer": Tensor(chunk_id)},
        context_ids=("turn-0",),
        context_prefix_tokens=3,
        raw_context_prefix_tokens=5,
        context_prefix_truncated_tokens=2,
    )


def test_cache_file_is_addressed_by_model_encoding_settings_sample_and_facts() -> None:
    settings = SimpleNamespace(
        model="Qwen/Qwen3-14B",
        kv_dtype="bfloat16",
        context_window="50",
        kv_max_position=32768.0,
        kv_block_size=True,
    )
    sample = ConversationSample(sample_id="conv-1", turns=[], qa=[], raw={})
    facts = [_fact("a"), _fact("b", text_hash="other", turn_id="turn-9")]
    hashed_facts = b"a\x00hash\x00turn-1\x00b\x00other\x00turn-9\x00"
    expected_facts_fingerprint = hashlib.sha256(hashed_facts).hexdigest()[:20]

    key = ChunkCacheKey.from_config(settings, sample, facts)

    assert CHUNK_CACHE_VERSION == 2
    assert facts_fingerprint(facts) == expected_facts_fingerprint
    assert facts_fingerprint([]) == hashlib.sha256(b"").hexdigest()[:20]
    assert key == ChunkCacheKey(
        model="Qwen/Qwen3-14B",
        dtype="bfloat16",
        context_window=50,
        max_position=32768,
        block_size=1,
        sample_id="conv-1",
        sample_fingerprint=sample_fingerprint(sample),
        facts_fingerprint=expected_facts_fingerprint,
    )
    directory = Path("/cache/kv-chunks/v2/Qwen__Qwen3-14B/bfloat16-w50-p32768-b1")
    assert key.directory("/cache") == directory
    assert key.path(Path("/cache")) == (
        directory / f"conv-1-{sample_fingerprint(sample)}-{expected_facts_fingerprint}.pt"
    )
    assert list(key.meta()) == [
        "model",
        "dtype",
        "context_window",
        "max_position",
        "block_size",
        "sample_id",
        "sample_fingerprint",
        "facts_fingerprint",
    ]


def test_saved_cache_holds_detached_cpu_tensors_for_the_facts_and_the_three_scaffold_slots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, log: list[tuple[str, str]]
) -> None:
    saved: list[tuple[Path, dict[str, Any]]] = []

    def save_torch_payload(path: Path, payload: dict[str, Any]) -> None:
        saved.append((path, payload))
        path.write_bytes(b"x" * 2_500_000)

    monkeypatch.setattr(chunk_cache, "save_torch_payload", save_torch_payload)
    meta = dict(META)
    path = tmp_path / "conv-1.pt"

    save_sample_chunks(
        path,
        meta=meta,
        fact_chunks={"a": _chunk("a", [1, 2]), "b": _chunk("b", [3])},
        scaffold_chunks={
            "footer": _chunk("footer", [6]),
            "header": _chunk("header", [4]),
            "not a scaffold slot": _chunk("other", [7]),
        },
        cos_table=Tensor("cos"),
        sin_table=Tensor("sin"),
    )

    [(saved_path, payload)] = saved
    assert saved_path == path
    assert list(payload) == [
        "version",
        "meta",
        "cos_table",
        "sin_table",
        "fact_chunks",
        "scaffold_chunks",
    ]
    assert payload["version"] == CHUNK_CACHE_VERSION
    assert payload["meta"] == META
    assert payload["meta"] is not meta
    for table in (payload["cos_table"], payload["sin_table"]):
        assert (table.device, table.detached) == ("cpu", True)
    assert list(payload["fact_chunks"]) == ["a", "b"]
    assert list(payload["scaffold_chunks"]) == ["header", "empty_memory", "footer"]
    assert payload["scaffold_chunks"]["empty_memory"] is None
    stored_fact = payload["fact_chunks"]["a"]
    assert {name: value for name, value in stored_fact.items() if name != "kv_by_layer"} == {
        "token_ids": [1, 2],
        "context_turn_ids": ["turn-0"],
        "context_prefix_tokens": 3,
        "raw_context_prefix_tokens": 5,
        "context_prefix_truncated_tokens": 2,
    }
    stored_tensor = stored_fact["kv_by_layer"]["layer"]
    assert (stored_tensor.name, stored_tensor.device, stored_tensor.detached) == ("a", "cpu", True)
    assert payload["scaffold_chunks"]["footer"]["token_ids"] == [6]
    assert payload_is_valid(payload, expected_meta=META, expected_fact_ids=["a", "b"])
    assert log == [("INFO", f"Saved KV chunk cache {path} (2 fact chunks, 2.5 MB)")]
