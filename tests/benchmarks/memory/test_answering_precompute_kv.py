from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from library_fakes import FakeTokenizer
from memory_config import make_memory_config, memory_runtime

from benchmarks.memory import precompute_kv
from benchmarks.memory.chunk_cache import ChunkCacheKey, load_sample_chunks
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, Turn
from benchmarks.memory.facts import unique_memory_facts
from benchmarks.memory.mem0.fact_catalog import MemoryFact, fact_catalog_hits
from benchmarks.memory.prompting import extract_memory_scaffold_token_ids
from inferscale.v1.types import EncodingPlan, KVChunk

torch = pytest.importorskip("torch")

LAYER = "model.layers.0.self_attn.attn"


class Encoder:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.device = settings["device"]
        self.max_position = settings["max_position"]
        self.tokenizer = FakeTokenizer()
        self.cos_table = torch.ones(4, 2)
        self.sin_table = torch.zeros(4, 2)
        self.closed = False

    def encode(self, chunk_id: str, token_ids: list[int]) -> KVChunk:
        return KVChunk(chunk_id=chunk_id, token_ids=list(token_ids), kv_by_layer=_kv(token_ids))

    def encode_plan(self, plan: EncodingPlan) -> KVChunk:
        return KVChunk(
            chunk_id=plan.chunk_id,
            token_ids=list(plan.target_token_ids),
            kv_by_layer=_kv(plan.target_token_ids),
            context_ids=plan.context_ids,
            context_prefix_tokens=len(plan.context_token_ids),
        )

    def close(self) -> None:
        self.closed = True


def _kv(token_ids: list[int]) -> dict[str, Any]:
    tokens = torch.tensor(token_ids, dtype=torch.float32).reshape(1, -1, 1, 1)
    return {LAYER: tokens.repeat(2, 1, 1, 1)}


class Workspace:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.samples = [_sample("s1"), _sample("s2")]
        monkeypatch.setattr(precompute_kv, "load_locomo", self._load_locomo)
        monkeypatch.setattr(precompute_kv, "fact_catalog_store_for", lambda config: self)
        monkeypatch.setattr(precompute_kv, "ChunkedRopeEncoder", self._encoder)

    def _load_locomo(self, path: Path, max_samples: int | None = None) -> list[ConversationSample]:
        return self.samples if max_samples is None else self.samples[:max_samples]

    def _encoder(self, **settings: Any) -> Encoder:
        return Encoder(settings)

    def path_for(self, sample: ConversationSample) -> Path:
        path = self.tmp_path / "catalogs" / f"{sample.sample_id}.json"
        path.parent.mkdir(exist_ok=True)
        path.touch()
        return path

    def load(self, sample: ConversationSample) -> tuple[MemoryFact, ...]:
        return _catalog(sample.sample_id)

    def config(self) -> MemoryRunConfig:
        settings: dict[str, Any] = {
            "runtime": memory_runtime(storage={"runtime_root": str(self.tmp_path)}),
            "dataset_path": self.tmp_path / "locomo.json",
            "context_window": 1,
            "inferscale": {
                "engine": {"block_size": 32},
                "kv": {"device": "cpu", "max_position": 2048, "dtype": "float16"},
            },
        }
        return make_memory_config(**settings)


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Workspace:
    return Workspace(monkeypatch, tmp_path)


def _sample(sample_id: str) -> ConversationSample:
    turns = [
        Turn(
            sample_id=sample_id,
            session_id="session_1",
            session_index=1,
            turn_index=turn_index,
            speaker="Alice",
            text=f"I like drink {turn_index}.",
        )
        for turn_index in range(2)
    ]
    return ConversationSample(sample_id=sample_id, turns=turns, qa=[], raw={})


def _catalog(sample_id: str) -> tuple[MemoryFact, ...]:
    return tuple(
        MemoryFact(
            id=f"{sample_id}-fact-{index}",
            text=f"Alice likes drink {index}.",
            created_at="2026-01-02T00:00:00+00:00",
            timestamp_epoch=1767312000,
            sample_id=sample_id,
            source_session_index=1,
            source_session_id="session_1",
            source_turn_index=index,
            source_turn_id=f"{sample_id}:session_1:{index}",
            speaker="Alice",
            role="user",
        )
        for index in range(2)
    )


def test_saved_chunks_hold_the_facts_encoded_after_their_context_and_the_prompt_scaffold(
    workspace: Workspace,
) -> None:
    config = workspace.config()
    sample = workspace.samples[0]

    precompute_kv.precompute_kv_chunks(config)

    facts = unique_memory_facts(fact_catalog_hits(_catalog("s1")))
    key = ChunkCacheKey.from_config(config, sample, facts)
    cached = load_sample_chunks(
        key.path(config.cache_root),
        device="cpu",
        expected_meta=key.meta(),
        expected_fact_ids=["s1-fact-0", "s1-fact-1"],
    )
    assert cached is not None
    tokenizer = FakeTokenizer()
    scaffold = extract_memory_scaffold_token_ids(tokenizer, block_size=32)
    assert cached.fact_chunks["s1-fact-1"].token_ids == tokenizer.encode(
        "Alice likes drink 1.\n", add_special_tokens=False
    )
    assert cached.fact_chunks["s1-fact-0"].context_ids == ()
    assert cached.fact_chunks["s1-fact-1"].context_ids == ("s1-fact-0",)
    assert {slot: chunk.token_ids for slot, chunk in cached.scaffold_chunks.items()} == {
        "header": scaffold.header_token_ids,
        "empty_memory": scaffold.empty_token_ids,
        "footer": scaffold.footer_token_ids,
    }
    assert torch.equal(cached.cos_table, torch.ones(4, 2))
    assert torch.equal(cached.sin_table, torch.zeros(4, 2))
