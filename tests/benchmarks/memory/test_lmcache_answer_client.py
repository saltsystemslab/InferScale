from __future__ import annotations

from typing import Any

import pytest
from library_fakes import FakeEngine, FakeTokenizer
from loguru import logger
from memory_config import make_memory_config

from benchmarks.common.clients import ChatResult
from benchmarks.common.config import ConfigError
from benchmarks.common.lmcache import runtime
from benchmarks.common.lmcache.runtime import WarmupStats
from benchmarks.common.lmcache.segments import blend_memory_layout, separator_token_ids
from benchmarks.memory import answer_lmcache, prediction
from benchmarks.memory.answer_lmcache import LMCacheBlendAnswerClient
from benchmarks.memory.data import ConversationSample, QuestionAnswer, Turn
from benchmarks.memory.mem0.fact_catalog import MemoryFact, fact_catalog_hits
from benchmarks.memory.prompting import (
    build_memory_query,
    extract_memory_scaffold_token_ids,
    select_memory_facts,
)
from benchmarks.memory.runtime_clients import MemoryWarmupClient, RuntimeClients, build_clients
from benchmarks.memory.setup_metrics import SAMPLE_SETUP_COLUMNS, record_memory_warmup
from benchmarks.memory.sweeps import MemoryCell, kv_context_windows

SEPARATOR = " # # "
KV_SWEEP = {
    "only-kv": {
        "top_k": [5],
        "variants": [{"name": "kv", "answer_backend": "kv-injection", "vector_backend": "jasper"}],
    }
}


def _lmcache_config(**values: Any):
    return make_memory_config(
        answer_backend="lmcache-blend", vector_backend="qdrant", skip_judge=True, **values
    )


# --- configuration -----------------------------------------------------------


def test_llama_config_accepts_the_arm_and_records_its_section() -> None:
    config = _lmcache_config()

    assert config.lmcache_settings().max_local_cpu_size_gb == 20
    saved = config.to_jsonable()["lmcache"]
    assert saved["vllm_prefix_caching"] is False
    assert saved["blend_special_str"] == SEPARATOR
    assert config.engine_config().enable_prefix_caching is True


def test_the_section_is_required_when_the_arm_is_used() -> None:
    with pytest.raises(ConfigError, match="lmcache is required"):
        _lmcache_config(lmcache=None, sweeps=KV_SWEEP)
    with pytest.raises(ConfigError, match="lmcache is required"):
        make_memory_config(answer_backend="kv-injection", lmcache=None)  # the lmcache sweep uses it


def test_runs_without_the_arm_need_no_section() -> None:
    config = make_memory_config(answer_backend="kv-injection", lmcache=None, sweeps=KV_SWEEP)

    assert config.lmcache is None
    assert config.to_jsonable()["lmcache"] is None
    with pytest.raises(RuntimeError, match="no lmcache section"):
        config.lmcache_settings()


def test_the_arm_runs_at_context_window_zero_only() -> None:
    with pytest.raises(ConfigError, match="context_window must be 0"):
        _lmcache_config(context_window=5)
    sweep = {
        "windows": {
            "top_k": [5],
            "variants": [
                {
                    "name": "lmcache-blend",
                    "answer_backend": "lmcache-blend",
                    "vector_backend": "qdrant",
                    "context_window": [0, 5],
                }
            ],
        }
    }
    with pytest.raises(ConfigError, match=r"context_window must be \[0\]"):
        _lmcache_config(sweeps=sweep)


def test_sweep_cell_names_the_arm_and_precompute_ignores_it() -> None:
    config = _lmcache_config()
    cell = MemoryCell(
        variant="lmcache-blend",
        answer_backend="lmcache-blend",
        vector_backend="qdrant",
        top_k=5,
        context_window=0,
    )

    assert cell.run_id("llama", 10, "STAMP") == "llama-lmcache-blend-mem0-qdrant10-k5-s0-STAMP"
    assert [(cell.variant, cell.top_k) for cell in config.sweep_cells("lmcache")] == [
        ("lmcache-blend", 50)
    ]
    assert kv_context_windows(config.sweeps.values(), 0) == (0, 5, 20, 50)


# --- client dispatch ---------------------------------------------------------


def test_build_clients_dispatches_the_arm_and_rejects_unknown_backends(monkeypatch: Any) -> None:
    monkeypatch.setattr(answer_lmcache, "VLLMEngine", lambda *_args, **_kwargs: FakeEngine())
    config = _lmcache_config()

    clients = build_clients(config)

    assert isinstance(clients.answer_client, LMCacheBlendAnswerClient)
    assert isinstance(clients.answer_client, MemoryWarmupClient)
    assert clients.judge_client is None

    config.answer_backend = "unknown"
    with pytest.raises(ValueError, match="Unsupported answer backend"):
        build_clients(config)


# --- the client against fakes ------------------------------------------------


class _Runtime:
    """Replaces the LMCache runtime helpers the client imports; records every call."""

    def __init__(self, monkeypatch: Any, *, matched_tokens: int | None = None) -> None:
        self.events: list[tuple[str, Any]] = []
        self.matched_tokens = matched_tokens
        self.separator = separator_token_ids(FakeTokenizer(), SEPARATOR)
        monkeypatch.setattr(answer_lmcache, "VLLMEngine", lambda *_a, **_k: FakeEngine())
        monkeypatch.setattr(answer_lmcache, "register_model", self._record("register_model"))
        monkeypatch.setattr(answer_lmcache, "lmcache_separator", self._separator)
        monkeypatch.setattr(answer_lmcache, "kv_bytes_per_token", lambda: 1000)
        monkeypatch.setattr(answer_lmcache, "clear_cache", self._record("clear_cache", 0))
        monkeypatch.setattr(answer_lmcache, "require_cache_capacity", self._capacity)
        monkeypatch.setattr(answer_lmcache, "warm_prompts", self._warm)
        monkeypatch.setattr(answer_lmcache, "shutdown", self._record("shutdown"))
        monkeypatch.setattr(answer_lmcache, "reset_load_stats", self._record("reset_load_stats"))
        monkeypatch.setattr(answer_lmcache, "snapshot_load_stats", self._snapshot)
        monkeypatch.setattr(runtime, "cpu_cache_usage_bytes", lambda: 3 * 1024 * 1024)

    def _record(self, name: str, result: Any = None) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            self.events.append((name, kwargs or args))
            return result

        return call

    def _separator(self) -> list[int]:
        self.events.append(("separator", ()))
        return self.separator

    def _capacity(self, **kwargs: Any) -> int:
        self.events.append(("capacity", kwargs))
        return 0

    def _warm(self, llm: Any, prompts: Any, *, batch_size: int) -> WarmupStats:
        self.events.append(("warm", [list(prompt) for prompt in prompts]))
        return WarmupStats(len(prompts), sum(len(p) for p in prompts), 7.0)

    def _snapshot(self) -> dict[str, Any]:
        self.events.append(("snapshot", None))
        return {"tokens_matched": self.matched_tokens, "load_time_ms": 3.0, "matched_tokens": {}}


def _sample(question_count: int = 2) -> ConversationSample:
    questions = [
        QuestionAnswer(
            sample_id="sample-1",
            question_id=f"q{index}",
            question=f"Question {index}?",
            answer="tea",
            category="1",
        )
        for index in range(1, question_count + 1)
    ]
    turn = Turn(
        sample_id="sample-1",
        session_id="session_1",
        session_index=1,
        turn_index=0,
        speaker="Alice",
        text="I like tea.",
        timestamp="2026-01-02",
    )
    return ConversationSample(sample_id="sample-1", turns=[turn], qa=questions, raw={})


def _fact(index: int) -> MemoryFact:
    return MemoryFact(
        id=f"fact-{index}",
        text=f"Alice likes drink {index}.",
        created_at="2026-01-02T00:00:00+00:00",
        timestamp_epoch=1767312000,
        sample_id="sample-1",
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=index,
        source_turn_id=f"sample-1:session_1:{index}",
        speaker="Alice",
        role="user",
    )


def _started_client(fakes: _Runtime, **values: Any) -> LMCacheBlendAnswerClient:
    client = LMCacheBlendAnswerClient(_lmcache_config(**values))
    client.start_llm()
    return client


def test_start_registers_the_model_and_turns_prefix_caching_off(monkeypatch: Any) -> None:
    fakes = _Runtime(monkeypatch)

    client = _started_client(fakes)

    engine: FakeEngine = client._engine  # noqa: SLF001
    assert engine.started_with is not None
    assert engine.started_with["kv_connector"] == "BlendConnector"
    assert engine.started_with["kv_connector_extra_config"]["lmcache.enable_blending"] is True
    assert client.config.engine_config().enable_prefix_caching is True
    assert [name for name, _ in fakes.events] == ["register_model", "separator"]
    assert fakes.events[0][1] == (engine.llm_instance,)


def test_warm_up_clears_the_cache_then_stores_header_and_every_fact(monkeypatch: Any) -> None:
    fakes = _Runtime(monkeypatch)
    client = _started_client(fakes)
    sample = _sample(question_count=3)
    hits = fact_catalog_hits([_fact(1), _fact(2)])

    metrics = client.warm_sample_memory(sample, hits)

    tokenizer = FakeTokenizer()
    scaffold = extract_memory_scaffold_token_ids(tokenizer, block_size=client.config.kv_block_size)
    facts = select_memory_facts(tokenizer, hits)
    names = [name for name, _ in fakes.events]
    assert names[-3:] == ["clear_cache", "capacity", "warm"]
    assert fakes.events[-1][1] == [scaffold.header_token_ids, *facts.ordered_token_ids()]
    capacity = fakes.events[-2][1]
    assert capacity["warm_tokens"] == sum(len(prompt) for prompt in fakes.events[-1][1])
    assert capacity["tail_tokens"] == runtime.TAIL_ALLOWANCE_TOKENS * 3
    assert capacity["bytes_per_token"] == 1000
    assert capacity["max_local_cpu_size_gb"] == 20
    assert metrics == {
        "lmcache_warm_time_ms": 7.0,
        "lmcache_warm_prompts": 3,
        "lmcache_warm_tokens": capacity["warm_tokens"],
        "lmcache_cpu_cache_mb": 3.0,
    }


def test_answer_lays_facts_between_separators_and_reports_lmcache_metrics(
    monkeypatch: Any,
) -> None:
    sample = _sample()
    hits = fact_catalog_hits([_fact(1), _fact(2)])
    tokenizer = FakeTokenizer()
    query = build_memory_query(
        tokenizer,
        sample.qa[0],
        block_size=16,
        max_position=32768,
        max_model_len=32768,
        max_answer_tokens=512,
    )
    facts = select_memory_facts(tokenizer, hits)
    layout = blend_memory_layout(
        query.scaffold, facts.ordered_token_ids(), separator_token_ids(tokenizer, SEPARATOR)
    )
    fakes = _Runtime(monkeypatch, matched_tokens=layout.expected_matched_tokens)
    client = _started_client(fakes)
    client.prepare_sample(sample)

    result = client.answer_with_retrieved_memory(
        sample=sample, qa=sample.qa[0], hits=hits, max_tokens=512, temperature=0.0, top_p=1.0
    )

    engine: FakeEngine = client._engine  # noqa: SLF001
    assert engine.prompts == [[*layout.token_ids, *query.query_tokens.token_ids]]
    assert [name for name, _ in fakes.events][-2:] == ["reset_load_stats", "snapshot"]
    metrics = result.metrics
    assert metrics["kv_memory_tokens"] == len(layout.token_ids)
    assert metrics["kv_query_tokens"] == len(query.query_tokens.token_ids)
    assert metrics["kv_selected_fact_ids"] == ["fact-2", "fact-1"]
    assert metrics["kv_prefix_caching"] == 0
    assert metrics["kv_context_window"] == 0
    assert metrics["lmcache_separator_tokens"] == len(layout.separator_token_ids)
    assert metrics["lmcache_segment_count"] == 3
    assert metrics["lmcache_expected_matched_tokens"] == layout.expected_matched_tokens
    assert metrics["lmcache_matched_tokens"] == layout.expected_matched_tokens
    assert metrics["lmcache_load_time_ms"] == 3.0
    assert metrics["lmcache_engine_time_to_first_token_ms"] == 12.0
    assert metrics["memory_token_budget"] == query.memory_token_budget
    assert result.ttft_ms == 12.0


def test_answer_warns_when_lmcache_matched_fewer_tokens_than_expected(monkeypatch: Any) -> None:
    fakes = _Runtime(monkeypatch, matched_tokens=1)
    client = _started_client(fakes)
    sample = _sample()
    client.prepare_sample(sample)
    warnings: list[str] = []
    handle = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        client.answer_with_retrieved_memory(
            sample=sample,
            qa=sample.qa[0],
            hits=fact_catalog_hits([_fact(1)]),
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
    finally:
        logger.remove(handle)

    assert any("matched 1 of" in message for message in warnings)


def test_answer_refuses_facts_that_exceed_the_memory_budget(monkeypatch: Any) -> None:
    fakes = _Runtime(monkeypatch)
    client = _started_client(fakes, inferscale={**_library_section(), "kv": _tiny_kv_section()})
    sample = _sample()
    client.prepare_sample(sample)

    with pytest.raises(RuntimeError, match="memory token budget"):
        client.answer_with_retrieved_memory(
            sample=sample,
            qa=sample.qa[0],
            hits=fact_catalog_hits([_fact(1), _fact(2)]),
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )


def test_answer_requires_start_and_the_prepared_sample(monkeypatch: Any) -> None:
    fakes = _Runtime(monkeypatch)
    client = LMCacheBlendAnswerClient(_lmcache_config())
    sample = _sample()
    with pytest.raises(RuntimeError, match="start_llm"):
        client.warm_sample_memory(sample, [])
    client.start_llm()
    with pytest.raises(RuntimeError, match="prepare_sample"):
        client.answer_with_retrieved_memory(
            sample=sample, qa=sample.qa[0], hits=[], max_tokens=8, temperature=0.0, top_p=1.0
        )

    client.close()

    assert ("shutdown", ()) in fakes.events
    assert client._engine.closed  # noqa: SLF001


def _library_section() -> dict[str, Any]:
    from memory_config import memory_config_data

    return memory_config_data()["inferscale"]


def _tiny_kv_section() -> dict[str, Any]:
    return {**_library_section()["kv"], "max_position": 48}


# --- the run stage's warm-up hook ---------------------------------------------


def test_setup_row_records_the_warm_up_before_the_prepare_time() -> None:
    columns = list(SAMPLE_SETUP_COLUMNS)
    warm_columns = [column for column in columns if column.startswith("lmcache_")]

    assert warm_columns == [
        "lmcache_warm_time_ms",
        "lmcache_warm_prompts",
        "lmcache_warm_tokens",
        "lmcache_cpu_cache_mb",
    ]
    assert columns.index("lmcache_cpu_cache_mb") < columns.index("answer_prepare_sample_time_ms")
    row: dict[str, Any] = {}
    record_memory_warmup(row, {"lmcache_warm_time_ms": "7.5", "lmcache_warm_prompts": 3})
    assert row == {
        "lmcache_warm_time_ms": 7.5,
        "lmcache_warm_prompts": 3,
        "lmcache_warm_tokens": None,
        "lmcache_cpu_cache_mb": None,
    }


def test_prediction_warms_a_warmup_client_after_preparing_the_sample(
    monkeypatch: Any, tmp_path: Any
) -> None:
    events: list[str] = []
    sample = _sample(question_count=1)
    fact = _fact(1)

    class FakeRetriever:
        def search(self, query: str, *, top_k: int) -> tuple[list, Any]:
            events.append("search")
            from benchmarks.common.vector_types import RetrievalMetrics

            return [fact.to_search_hit(1)], RetrievalMetrics(1.0, 2.0, 3.0)

        def log_cache_stats(self) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeMemoryBuilder:
        def __init__(self, config: Any) -> None:
            pass

        def load_fact_catalog(self, selected_sample: ConversationSample) -> tuple[MemoryFact, ...]:
            return (fact,)

        def build_retriever_with_metrics(self, selected_sample: Any) -> tuple[Any, dict]:
            return FakeRetriever(), {"memory_setup_time_ms": 4.0}

    class FakeWarmupClient:
        def start_llm(self) -> None:
            events.append("start_llm")

        def prepare_sample(self, selected_sample: ConversationSample) -> None:
            events.append("prepare_sample")

        def warm_sample_memory(self, selected_sample: Any, facts: list) -> dict[str, Any]:
            events.append("warm")
            assert [hit.id for hit in facts] == ["fact-1"]
            return {"lmcache_warm_time_ms": 9.0, "lmcache_warm_tokens": 42}

        def answer_with_retrieved_memory(self, **_: Any) -> ChatResult:
            events.append("answer")
            return ChatResult(content="tea", ttft_ms=6.0)

        def close_sample(self) -> None:
            events.append("close_sample")

    monkeypatch.setattr(prediction, "load_locomo", lambda *_a, **_k: [sample])
    monkeypatch.setattr(prediction, "SampleMemoryBuilder", FakeMemoryBuilder)
    config = _lmcache_config(
        dataset_path=tmp_path / "unused.json",
        results_dir=tmp_path / "results",
        run_id="warmup-hook",
        top_k=1,
        max_samples=1,
        log_every=0,
    )

    result = prediction.run_prediction_mode(
        config, RuntimeClients(answer_client=FakeWarmupClient(), judge_client=None)
    )

    assert events == ["start_llm", "prepare_sample", "warm", "search", "answer", "close_sample"]
    setup_row = result.sample_setup_metrics[0]
    assert setup_row["lmcache_warm_time_ms"] == 9.0
    assert setup_row["lmcache_warm_tokens"] == 42
    assert setup_row["lmcache_warm_prompts"] is None
    assert setup_row["answer_prepare_sample_time_ms"] >= 0
    assert setup_row["sample_setup_time_ms"] >= 4.0
