from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from library_fakes import FakeTokenizer

from benchmarks.common.lmcache import runtime
from benchmarks.common.lmcache.config import LMCacheConfig
from benchmarks.common.lmcache.runtime import WarmupStats
from benchmarks.common.lmcache.segments import separator_token_ids
from benchmarks.rag import answer_lmcache
from benchmarks.rag.answer_lmcache import RagLMCacheBlendAnswerClient
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import RagChunk, RagPromptProfile, RagQuery
from inferscale.v1.types import SearchHit

CHUNKS = [
    RagChunk(chunk_id="c1", doc_id="d", chunk_index=0, token_ids=[101, 102, 103], text="abc"),
    RagChunk(chunk_id="c2", doc_id="d", chunk_index=1, token_ids=[201, 202], text="de"),
]
QUERY = RagQuery(
    query_id="q1", question="What?", gold_answers=("x",), question_type="t", evidence=()
)
HITS = [
    SearchHit(id="c1", payload={}, score=0.9, distance=0.1, rank=1),
    SearchHit(id="c2", payload={}, score=0.8, distance=0.2, rank=2),
]


@pytest.fixture
def blending(monkeypatch, engine):
    events: list[Any] = []
    load_stats = {"tokens_matched": 10**6, "load_time_ms": 3}

    def warm_prompts(llm, prompts, *, batch_size: int) -> WarmupStats:
        events.append(("warm", len(prompts), batch_size))
        return WarmupStats(len(prompts), sum(len(prompt) for prompt in prompts), 1.5)

    def snapshot_load_stats() -> dict[str, Any]:
        events.append("load stats read")
        return load_stats

    generate = engine.generate

    def generate_and_record(prompt_token_ids, **sampling):
        events.append("generate")
        return generate(prompt_token_ids, **sampling)

    monkeypatch.setattr(engine, "generate", generate_and_record)
    monkeypatch.setattr(answer_lmcache, "register_model", lambda llm: events.append("register"))
    monkeypatch.setattr(
        answer_lmcache, "lmcache_separator", lambda: separator_token_ids(FakeTokenizer(), " # # ")
    )
    monkeypatch.setattr(answer_lmcache, "kv_bytes_per_token", lambda: 2)
    monkeypatch.setattr(runtime, "cpu_cache_usage_bytes", lambda: 4096)
    monkeypatch.setattr(answer_lmcache, "warm_prompts", warm_prompts)
    monkeypatch.setattr(
        answer_lmcache, "reset_load_stats", lambda: events.append("load stats reset")
    )
    monkeypatch.setattr(answer_lmcache, "snapshot_load_stats", snapshot_load_stats)
    monkeypatch.setattr(answer_lmcache, "shutdown", lambda: events.append("lmcache shut down"))
    monkeypatch.setattr(
        answer_lmcache, "empty_cuda_cache", lambda **options: events.append(("empty", options))
    )
    config = RagBenchConfig(
        answer_backend="lmcache-blend", lmcache=LMCacheConfig(max_local_cpu_size_gb=1)
    )
    client = RagLMCacheBlendAnswerClient(
        config,
        chunks=CHUNKS,
        prompt_profile=RagPromptProfile(
            system_prompt="Use the passages.\n", answer_instruction="Be brief.\n"
        ),
        query_count=3,
    )
    return SimpleNamespace(client=client, events=events, load_stats=load_stats, engine=engine)


def test_load_statistics_cover_exactly_one_generation(blending):
    blending.client.start_llm()
    blending.events.clear()

    answer = blending.client.answer(QUERY, HITS)

    assert blending.events == ["load stats reset", "generate", "load stats read"]
    assert answer.metrics["lmcache_load_time_ms"] == 3.0
    assert isinstance(answer.metrics["lmcache_load_time_ms"], float)
