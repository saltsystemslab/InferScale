from __future__ import annotations

from pathlib import Path
import pytest

from locomo_jasper_bench.throughput.config import ThroughputConfig
from locomo_jasper_bench.throughput.reporting import RESULT_COLUMNS, validate_result_row
from locomo_jasper_bench.throughput.worker import (
    _build_warmup_prompts,
    _result_row,
    _select_chunks_for_fact_ids,
    run_condition,
)


def test_select_chunks_preserves_reverse_ranked_fact_order() -> None:
    chunks = {"fact-a": "chunk-a", "fact-b": "chunk-b", "fact-c": "chunk-c"}

    selected = _select_chunks_for_fact_ids(["fact-c", "fact-a"], chunks)

    assert selected == ["chunk-c", "chunk-a"]


def test_select_chunks_rejects_missing_chunk_or_empty_retrieval() -> None:
    with pytest.raises(RuntimeError, match="no pre-encoded KV chunk"):
        _select_chunks_for_fact_ids(["fact-z"], {"fact-a": "chunk-a"})
    with pytest.raises(RuntimeError, match="no facts"):
        _select_chunks_for_fact_ids([], {"fact-a": "chunk-a"})


def _config(tmp_path: Path) -> ThroughputConfig:
    return ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
        run_id="worker-test",
    )


def test_kv_result_row_matches_report_schema(tmp_path: Path) -> None:
    row = _result_row(
        _config(tmp_path),
        10,
        condition="kv_injection",
        vector_backend="jasper",
        jasper_effective_beam_width=64,
        fact_count=212.5,
        generation_time_s=1.0,
        retrieval_time_s=0.5,
        vector_search_time_s=0.1,
        prompt_build_time_s=0.2,
        memory_setup_time_s=3.0,
        kv_precompute_time_s=8.0,
        kv_compose_time_s=0.3,
        kv_verify_time_s=0.05,
        engine_startup_time_s=20.0,
        kv_store_gpu_mb=64.0,
        kv_requests_loaded=20,
        total_input_tokens=10000,
        total_output_tokens=1000,
    )

    assert tuple(row) == RESULT_COLUMNS
    assert row["throughput_qps"] == 20.0  # 20 requests / 1s generation
    assert row["fact_count"] == 212.5
    assert row["kv_verify_time_s"] == 0.05
    assert row["kv_requests_loaded"] == 20
    assert validate_result_row(row) == validate_result_row(dict(row))


def test_result_row_qps_counts_only_generation_time(tmp_path: Path) -> None:
    row = _result_row(
        _config(tmp_path),
        10,
        condition="mem0_jasper",
        vector_backend="jasper",
        generation_time_s=2.0,
        retrieval_time_s=50.0,
        prompt_build_time_s=5.0,
        total_input_tokens=1000,
        total_output_tokens=500,
    )

    assert row["throughput_qps"] == pytest.approx(10.0)
    assert row["avg_latency_ms"] == pytest.approx(100.0)
    assert row["kv_requests_loaded"] == 0


def test_result_row_rejects_non_positive_generation_time(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="greater than zero"):
        _result_row(
            _config(tmp_path),
            10,
            condition="no_memory",
            generation_time_s=0.0,
            total_input_tokens=1,
            total_output_tokens=1,
        )


def test_run_condition_rejects_removed_prompt_injection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported condition"):
        run_condition(_config(tmp_path), "prompt_injection", (2,))


def test_run_condition_requires_single_count_kv_worker(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one user count"):
        run_condition(_config(tmp_path), "kv_injection", (2, 3))


def _prompts(lengths: list[int]) -> list[dict[str, list[int]]]:
    return [{"prompt_token_ids": list(range(length))} for length in lengths]


def test_warmup_prompts_match_lengths_and_stay_in_vocab() -> None:
    prompts = _prompts([5, 12, 1])
    warmup = _build_warmup_prompts(prompts, vocab_size=100, seed=42)

    assert [len(p["prompt_token_ids"]) for p in warmup] == [5, 12, 1]
    assert all(
        0 <= token < 100 for p in warmup for token in p["prompt_token_ids"]
    )


def test_warmup_prompts_are_deterministic_but_not_the_real_prompts() -> None:
    prompts = _prompts([64, 64])
    first = _build_warmup_prompts(prompts, vocab_size=32000, seed=7)
    second = _build_warmup_prompts(prompts, vocab_size=32000, seed=7)

    assert first == second
    assert all(
        warm["prompt_token_ids"] != real["prompt_token_ids"]
        for warm, real in zip(first, prompts)
    )


def test_warmup_prompts_cap_at_ten() -> None:
    warmup = _build_warmup_prompts(_prompts([4] * 25), vocab_size=100, seed=0)

    assert len(warmup) == 10


def test_warmup_prompts_reject_empty_vocab() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        _build_warmup_prompts(_prompts([4]), vocab_size=0, seed=0)


def test_kv_warmup_prompt_extends_memory_with_random_tail() -> None:
    from locomo_jasper_bench.throughput.worker import _build_kv_warmup_prompt

    memory = [11, 12, 13]
    prompt = _build_kv_warmup_prompt(memory, vocab_size=100, seed=42)

    tokens = prompt["prompt_token_ids"]
    assert tokens[: len(memory)] == memory
    assert len(tokens) > len(memory)
    assert all(0 <= token < 100 for token in tokens[len(memory):])
    assert prompt == _build_kv_warmup_prompt(memory, vocab_size=100, seed=42)


class _FakeWarmupLLM:
    def __init__(self, *, has_reset: bool = True, reset_result: object = True) -> None:
        self.generate_calls = 0
        self.reset_calls = 0
        self._reset_result = reset_result
        if not has_reset:
            self.reset_prefix_cache = None  # type: ignore[assignment]

    def get_tokenizer(self):
        return type("Tok", (), {"vocab_size": 100})()

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.generate_calls += 1

    def reset_prefix_cache(self):
        self.reset_calls += 1
        return self._reset_result


def test_warmup_zero_batches_still_resets_prefix_cache() -> None:
    from locomo_jasper_bench.throughput.worker import _warm_up

    llm = _FakeWarmupLLM()
    _warm_up(llm, _prompts([4]), sampling_params=object(), batches=0, seed=1)

    assert llm.generate_calls == 0
    assert llm.reset_calls == 1


def test_warmup_reset_failure_raises() -> None:
    from locomo_jasper_bench.throughput.worker import _warm_up

    llm = _FakeWarmupLLM(reset_result=False)
    with pytest.raises(RuntimeError, match="reset_prefix_cache"):
        _warm_up(llm, _prompts([4]), sampling_params=object(), batches=1, seed=1)
    assert llm.generate_calls == 1


def test_warmup_missing_reset_api_raises() -> None:
    from locomo_jasper_bench.throughput.worker import _reset_prefix_cache

    with pytest.raises(RuntimeError, match="no reset_prefix_cache"):
        _reset_prefix_cache(object())
