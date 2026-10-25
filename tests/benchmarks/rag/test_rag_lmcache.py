"""The RAG LMCache CacheBlend arm: configuration, dispatch, warm-up, and the blend prompt."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from library_fakes import FakeEngine, FakeTokenizer

from benchmarks.common.config import ConfigError, RuntimeConfig
from benchmarks.common.lmcache import runtime
from benchmarks.common.lmcache.config import LMCacheConfig, lmcache_kv_transfer_config
from benchmarks.common.lmcache.runtime import WarmupStats
from benchmarks.common.lmcache.segments import blend_memory_layout, separator_token_ids
from benchmarks.rag import answer_lmcache, answer_run, engine_client
from benchmarks.rag.answer_lmcache import RagLMCacheBlendAnswerClient
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import RagChunk, RagPromptProfile, RagQuery
from benchmarks.rag.prompting import build_rag_query_tokens, extract_rag_scaffold_token_ids
from benchmarks.rag.results import QUERY_METRICS_COLUMNS, TIMING_METRIC_KEYS
from inferscale.v1.kv.compose import reverse_ranked_ids
from inferscale.v1.types import SearchHit

SEPARATOR = " # # "
PROFILE = RagPromptProfile(system_prompt="Use the passages.\n", answer_instruction="Be brief.\n")
CHUNKS = [
    RagChunk(chunk_id="c1", doc_id="d", chunk_index=0, token_ids=[101, 102, 103], text="abc"),
    RagChunk(chunk_id="c2", doc_id="d", chunk_index=1, token_ids=[201, 202], text="de"),
]
QUERY = RagQuery(
    query_id="q1", question="What?", gold_answers=("x",), question_type="t", evidence=()
)


@pytest.fixture
def custom_runtime(tmp_path, monkeypatch) -> RuntimeConfig:
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps({"storage": {"runtime_root": str(tmp_path)}}))
    monkeypatch.delenv("JUDGE_LLM_API_KEY", raising=False)
    return RuntimeConfig.from_json(path, root=tmp_path)


def _hits() -> list[SearchHit]:
    return [
        SearchHit(id="c1", payload={}, score=0.9, distance=0.1, rank=1),
        SearchHit(id="c2", payload={}, score=0.8, distance=0.2, rank=2),
    ]


def test_lmcache_backend_is_accepted_and_recorded(custom_runtime) -> None:
    config = RagBenchConfig.from_dict(
        {"answer_backend": "lmcache-blend", "lmcache": {"max_local_cpu_size_gb": 8}},
        runtime=custom_runtime,
    )

    assert config.result_mode() == "rag-lmcache-blend"
    assert config.lmcache_settings() == LMCacheConfig(max_local_cpu_size_gb=8.0)
    data = config.to_jsonable()
    assert data["mode"] == "rag-lmcache-blend"
    assert data["lmcache"]["vllm_prefix_caching"] is False
    assert data["lmcache"]["blend_check_layers"] == [1]


@pytest.mark.parametrize(
    "data",
    [
        {"answer_backend": "lmcache-blend"},
        {"sweeps": {"lmcache": {"answer_backends": ["lmcache-blend"]}}},
    ],
)
def test_lmcache_section_is_required_when_the_arm_is_used(custom_runtime, data) -> None:
    with pytest.raises(ConfigError, match="rag.lmcache is required"):
        RagBenchConfig.from_dict(data, runtime=custom_runtime)


def test_lmcache_section_is_optional_for_the_other_arms(custom_runtime) -> None:
    without = RagBenchConfig.from_dict({}, runtime=custom_runtime)
    with_section = RagBenchConfig.from_dict(
        {"lmcache": {"max_local_cpu_size_gb": 8}}, runtime=custom_runtime
    )

    assert without.lmcache is None and without.to_jsonable()["lmcache"] is None
    assert with_section.lmcache == LMCacheConfig(max_local_cpu_size_gb=8.0)
    with pytest.raises(RuntimeError, match="no lmcache section"):
        without.lmcache_settings()


def test_unknown_backend_has_no_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported answer backend"):
        RagBenchConfig(answer_backend="other").result_mode()


def test_build_answer_client_dispatches_every_backend(monkeypatch) -> None:
    corpus = SimpleNamespace(
        chunks=CHUNKS,
        chunks_by_id={chunk.chunk_id: chunk for chunk in CHUNKS},
        queries=[QUERY, QUERY, QUERY],
        docs=[],
        spec=SimpleNamespace(prompt_profile=PROFILE),
    )
    blend_factory = Mock()
    prompt_factory = Mock()
    monkeypatch.setattr(answer_lmcache, "RagLMCacheBlendAnswerClient", blend_factory)
    monkeypatch.setattr(
        "benchmarks.rag.answer_prompt_injection.RagPromptInjectionAnswerClient", prompt_factory
    )

    config = RagBenchConfig(answer_backend="lmcache-blend", max_queries=2)
    assert answer_run.build_answer_client(config, corpus) is blend_factory.return_value
    blend_factory.assert_called_once_with(
        config, chunks=CHUNKS, prompt_profile=PROFILE, query_count=2
    )
    answer_run.build_answer_client(RagBenchConfig(answer_backend="prompt-injection"), corpus)
    prompt_factory.assert_called_once()
    with pytest.raises(ValueError, match="Unsupported answer backend"):
        answer_run.build_answer_client(RagBenchConfig(answer_backend="other"), corpus)


def _client(monkeypatch, *, bytes_per_token: int = 2, **config_overrides):
    """A client over FakeEngine with the LMCache runtime helpers replaced by fakes."""
    engine = FakeEngine()
    engine_factory = Mock(return_value=engine)
    monkeypatch.setattr(engine_client, "VLLMEngine", engine_factory)
    monkeypatch.setattr(engine_client, "force_vllm_inprocess_mode", Mock())
    calls: list[str] = []
    monkeypatch.setattr(
        answer_lmcache, "register_model", lambda llm: calls.append(f"register:{id(llm)}")
    )
    monkeypatch.setattr(
        answer_lmcache,
        "lmcache_separator",
        lambda: separator_token_ids(FakeTokenizer(), SEPARATOR),
    )
    monkeypatch.setattr(answer_lmcache, "kv_bytes_per_token", lambda: bytes_per_token)
    monkeypatch.setattr(runtime, "cpu_cache_usage_bytes", lambda: 4096)
    warmed: list[list[list[int]]] = []

    def fake_warm(llm, prompts, *, batch_size):
        calls.append("warm")
        warmed.append([list(prompt) for prompt in prompts])
        return WarmupStats(len(prompts), sum(len(prompt) for prompt in prompts), 1.5)

    monkeypatch.setattr(answer_lmcache, "warm_prompts", fake_warm)
    monkeypatch.setattr(answer_lmcache, "shutdown", lambda: calls.append("shutdown"))
    config = RagBenchConfig(
        answer_backend="lmcache-blend",
        lmcache=LMCacheConfig(max_local_cpu_size_gb=1),
        **config_overrides,
    )
    client = RagLMCacheBlendAnswerClient(
        config, chunks=CHUNKS, prompt_profile=PROFILE, query_count=3
    )
    return client, engine, engine_factory, calls, warmed


def test_start_warms_header_and_every_chunk_on_an_engine_without_prefix_caching(
    monkeypatch,
) -> None:
    client, engine, engine_factory, calls, warmed = _client(monkeypatch)

    client.start_llm()

    assert engine_factory.call_args.args[2].enable_prefix_caching is False
    assert engine.started_with == lmcache_kv_transfer_config(LMCacheConfig(1))
    assert calls == [f"register:{id(engine.llm)}", "warm"]
    scaffold = extract_rag_scaffold_token_ids(
        FakeTokenizer(), system_prompt=PROFILE.system_prompt, block_size=16
    )
    assert warmed == [[scaffold.header_token_ids, [101, 102, 103], [201, 202]]]
    stats = client.setup_stats()
    assert stats["lmcache_warm_prompts"] == 3
    assert stats["lmcache_warm_tokens"] == len(scaffold.header_token_ids) + 5
    assert stats["lmcache_warm_time_ms"] == 1.5
    assert stats["lmcache_cpu_cache_mb"] == 4096 / (1024 * 1024)
    assert stats["lmcache_kv_bytes_per_token"] == 2
    assert stats["lmcache_separator_tokens"] == 5
    assert stats["lmcache_required_cpu_cache_gb"] == pytest.approx(
        2 * (stats["lmcache_warm_tokens"] + 128 * 3) / 1e9
    )
    assert stats["lmcache_max_local_cpu_size_gb"] == 1.0

    client.close()

    assert calls[-1] == "shutdown"
    assert engine.closed


def test_start_fails_and_closes_the_engine_when_the_corpus_does_not_fit(monkeypatch) -> None:
    client, engine, _, calls, warmed = _client(monkeypatch, bytes_per_token=10**9)

    with pytest.raises(RuntimeError, match="lmcache.max_local_cpu_size_gb"):
        client.start_llm()

    assert warmed == []
    assert engine.closed


def test_answer_sends_the_blend_layout_and_reports_lmcache_metrics(monkeypatch) -> None:
    client, engine, _, _, _ = _client(monkeypatch)
    client.start_llm()
    tokenizer = FakeTokenizer()
    scaffold = extract_rag_scaffold_token_ids(
        tokenizer, system_prompt=PROFILE.system_prompt, block_size=16
    )
    separator = separator_token_ids(tokenizer, SEPARATOR)
    ordered = reverse_ranked_ids(_hits())
    layout = blend_memory_layout(
        scaffold, [next(c.token_ids for c in CHUNKS if c.chunk_id == i) for i in ordered], separator
    )
    monkeypatch.setattr(
        answer_lmcache,
        "snapshot_load_stats",
        lambda: {"tokens_matched": layout.expected_matched_tokens, "load_time_ms": 3.0},
    )

    result = client.answer(QUERY, _hits(), query_started_at=99.0)

    query_tokens = build_rag_query_tokens(
        tokenizer, layout.token_ids, QUERY, answer_instruction=PROFILE.answer_instruction
    )
    assert engine.prompts == [[*layout.token_ids, *query_tokens.token_ids]]
    assert result.ttft_ms == 12.0
    metrics = result.metrics
    assert metrics["kv_memory_tokens"] == len(layout.token_ids)
    assert metrics["kv_query_tokens"] == len(query_tokens.token_ids)
    assert metrics["kv_selected_chunk_ids"] == ordered
    assert metrics["kv_prefix_caching"] == 0
    assert metrics["retrieved_chunk_count"] == 2
    assert metrics["lmcache_separator_tokens"] == len(separator)
    assert metrics["lmcache_segment_count"] == 3
    assert metrics["lmcache_expected_matched_tokens"] == layout.expected_matched_tokens
    assert metrics["lmcache_matched_tokens"] == layout.expected_matched_tokens
    assert metrics["lmcache_load_time_ms"] == 3.0
    assert metrics["lmcache_engine_time_to_first_token_ms"] == 12.0
    assert "query_to_first_token_ms" in metrics
    csv_keys = ("lmcache_matched_tokens", "lmcache_expected_matched_tokens", "lmcache_load_time_ms")
    for key in csv_keys:
        assert key in TIMING_METRIC_KEYS and key in QUERY_METRICS_COLUMNS
    assert "lmcache_engine_time_to_first_token_ms" in TIMING_METRIC_KEYS


def test_answer_rejects_a_memory_over_the_budget(monkeypatch) -> None:
    client, _, _, _, _ = _client(monkeypatch, kv_max_position=8)
    client.start_llm()

    with pytest.raises(RuntimeError, match="top_k"):
        client.answer(QUERY, _hits())


def test_answer_requires_a_started_engine(monkeypatch) -> None:
    client, _, _, _, _ = _client(monkeypatch)

    with pytest.raises(RuntimeError, match="start_llm"):
        client.answer(QUERY, _hits())
