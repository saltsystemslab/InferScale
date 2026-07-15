from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from locomo_jasper_bench.throughput.config import ThroughputConfig
from locomo_jasper_bench.throughput.engine import build_warmup_prompts
from locomo_jasper_bench.throughput.kv_condition import _select_chunks_for_fact_ids
from locomo_jasper_bench.throughput.projection import (
    check_kv_gpu_projection,
    check_pinned_host_projection,
    parse_mem_available_bytes,
)
from locomo_jasper_bench.throughput.reporting import (
    RESULT_COLUMNS,
    build_result_row,
    validate_result_row,
)
from locomo_jasper_bench.throughput.worker import run_condition


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


def test_mem_available_parser_reads_meminfo_kilobytes() -> None:
    text = "MemTotal:       1000 kB\nMemAvailable:   2048 kB\nSwapTotal: 0 kB\n"

    assert parse_mem_available_bytes(text) == 2048 * 1024
    assert parse_mem_available_bytes("MemTotal: 1000 kB\n") is None
    assert parse_mem_available_bytes("MemAvailable: garbage kB\n") is None


def test_pinned_host_projection_rejects_footprint_above_available_ram(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "locomo_jasper_bench.throughput.projection.available_host_memory_bytes",
        lambda: 10 * 2**30,
    )

    with pytest.raises(RuntimeError, match="pinned-host KV footprint exceeds available RAM"):
        check_pinned_host_projection(
            _config(tmp_path),
            composed_bytes=9.5 * 2**30,
            num_users=100,
            total_requests=200,
        )

    check_pinned_host_projection(
        _config(tmp_path),
        composed_bytes=8 * 2**30,
        num_users=100,
        total_requests=200,
    )


def test_pinned_host_projection_skips_without_a_memory_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "locomo_jasper_bench.throughput.projection.available_host_memory_bytes",
        lambda: None,
    )

    check_pinned_host_projection(
        _config(tmp_path),
        composed_bytes=float("inf"),
        num_users=1,
        total_requests=2,
    )


class _FakeChunk:
    """Duck-typed EncodedChunk: kv_by_layer values only need .nbytes."""

    def __init__(self, nbytes: int, num_tokens: int) -> None:
        self.kv_by_layer = {"layer": SimpleNamespace(nbytes=nbytes)}
        self.token_ids = list(range(num_tokens))


def _install_fake_cuda_device(monkeypatch: pytest.MonkeyPatch, total_memory: int) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda index: SimpleNamespace(total_memory=total_memory)
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _projection_kwargs(
    *,
    sample_bytes: int,
    sample_tokens: int,
    unique_sample_count: int,
    total_requests: int,
) -> dict:
    return {
        "num_users": total_requests // 2,
        "total_requests": total_requests,
        "unique_sample_count": unique_sample_count,
        "scaffold_chunks": (_FakeChunk(0, 0), _FakeChunk(0, 0)),
        "first_sample_chunks": {"fact-a": _FakeChunk(sample_bytes, sample_tokens)},
    }


def test_kv_gpu_projection_rejects_sources_plus_composed_in_setup_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_cuda_device(monkeypatch, total_memory=100 * 2**30)

    # 14 samples x 7GiB sources = 98GiB plus composed copies and graphs
    # exceeds the 97GiB cap before the vLLM pool even exists.
    with pytest.raises(RuntimeError, match="setup phase"):
        check_kv_gpu_projection(
            _config(tmp_path),
            **_projection_kwargs(
                sample_bytes=7 * 2**30,
                sample_tokens=1000,
                unique_sample_count=14,
                total_requests=2,
            ),
        )


def test_kv_gpu_projection_rejects_gpu_backend_composed_in_generate_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_cuda_device(monkeypatch, total_memory=100 * 2**30)
    config = _config(tmp_path)
    config.kv_store_backend = "gpu"
    config.kv_gpu_memory_utilization = 0.8

    # Pool 80GiB + 20 composed copies x 1GiB stays resident for the GPU
    # store, exceeding the cap only once the engine pool is up.
    with pytest.raises(RuntimeError, match="generate phase"):
        check_kv_gpu_projection(
            config,
            **_projection_kwargs(
                sample_bytes=1 * 2**30,
                sample_tokens=100,
                unique_sample_count=1,
                total_requests=20,
            ),
        )


def test_kv_gpu_projection_accepts_cpu_backend_at_scale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_cuda_device(monkeypatch, total_memory=100 * 2**30)
    monkeypatch.setattr(
        "locomo_jasper_bench.throughput.projection.available_host_memory_bytes",
        lambda: 200 * 2**30,
    )
    config = _config(tmp_path)
    config.kv_store_backend = "cpu"

    # The same 20-request workload passes with cpu: only the staging
    # slots stay device-resident during generation.
    check_kv_gpu_projection(
        config,
        **_projection_kwargs(
            sample_bytes=1 * 2**30,
            sample_tokens=100,
            unique_sample_count=1,
            total_requests=20,
        ),
    )


def test_kv_result_row_matches_report_schema(tmp_path: Path) -> None:
    row = build_result_row(
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
    row = build_result_row(
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
        build_result_row(
            _config(tmp_path),
            10,
            condition="mem0_jasper",
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
    warmup = build_warmup_prompts(prompts, vocab_size=100, seed=42)

    assert [len(p["prompt_token_ids"]) for p in warmup] == [5, 12, 1]
    assert all(
        0 <= token < 100 for p in warmup for token in p["prompt_token_ids"]
    )


def test_warmup_prompts_are_deterministic_but_not_the_real_prompts() -> None:
    prompts = _prompts([64, 64])
    first = build_warmup_prompts(prompts, vocab_size=32000, seed=7)
    second = build_warmup_prompts(prompts, vocab_size=32000, seed=7)

    assert first == second
    assert all(
        warm["prompt_token_ids"] != real["prompt_token_ids"]
        for warm, real in zip(first, prompts)
    )


def test_warmup_prompts_cap_at_ten() -> None:
    warmup = build_warmup_prompts(_prompts([4] * 25), vocab_size=100, seed=0)

    assert len(warmup) == 10


def test_warmup_prompts_reject_empty_vocab() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        build_warmup_prompts(_prompts([4]), vocab_size=0, seed=0)


def test_kv_warmup_prompt_extends_memory_with_random_tail() -> None:
    from locomo_jasper_bench.throughput.engine import build_kv_warmup_prompt

    memory = [11, 12, 13]
    prompt = build_kv_warmup_prompt(memory, vocab_size=100, seed=42)

    tokens = prompt["prompt_token_ids"]
    assert tokens[: len(memory)] == memory
    assert len(tokens) > len(memory)
    assert all(0 <= token < 100 for token in tokens[len(memory):])
    assert prompt == build_kv_warmup_prompt(memory, vocab_size=100, seed=42)


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
    from locomo_jasper_bench.throughput.engine import warm_up

    llm = _FakeWarmupLLM()
    warm_up(llm, _prompts([4]), sampling_params=object(), batches=0, seed=1)

    assert llm.generate_calls == 0
    assert llm.reset_calls == 1


def test_warmup_reset_failure_raises() -> None:
    from locomo_jasper_bench.throughput.engine import warm_up

    llm = _FakeWarmupLLM(reset_result=False)
    with pytest.raises(RuntimeError, match="reset_prefix_cache"):
        warm_up(llm, _prompts([4]), sampling_params=object(), batches=1, seed=1)
    assert llm.generate_calls == 1


def test_warmup_missing_reset_api_raises() -> None:
    from locomo_jasper_bench.throughput.engine import reset_prefix_cache

    with pytest.raises(RuntimeError, match="no reset_prefix_cache"):
        reset_prefix_cache(object())


class _FakeSamplingParams:
    def __init__(self, extra_args: dict | None = None) -> None:
        self.extra_args = extra_args

    def clone(self) -> "_FakeSamplingParams":
        return _FakeSamplingParams(
            dict(self.extra_args) if self.extra_args is not None else None
        )


def test_routed_sampling_params_carry_memory_user_ids() -> None:
    from locomo_jasper_bench.kv.connector_utils import MEMORY_USER_ID_EXTRA_ARG
    from locomo_jasper_bench.throughput.engine import sampling_params_with_memory_user_ids

    base = _FakeSamplingParams(extra_args={"existing": 1})
    routed = sampling_params_with_memory_user_ids(base, ["request-00000", "request-00001"])

    assert [params.extra_args[MEMORY_USER_ID_EXTRA_ARG] for params in routed] == [
        "request-00000",
        "request-00001",
    ]
    # Cloned per request: existing extra args preserved, base untouched.
    assert all(params.extra_args["existing"] == 1 for params in routed)
    assert MEMORY_USER_ID_EXTRA_ARG not in base.extra_args
    assert routed[0] is not routed[1]


def test_routed_sampling_params_require_clone() -> None:
    from locomo_jasper_bench.throughput.engine import sampling_params_with_memory_user_ids

    with pytest.raises(RuntimeError, match="clone"):
        sampling_params_with_memory_user_ids(object(), ["request-00000"])


def test_extract_user_id_reads_sampling_extra_args() -> None:
    from types import SimpleNamespace

    from locomo_jasper_bench.kv.connector_utils import (
        MEMORY_USER_ID_EXTRA_ARG,
        extract_user_id,
    )

    request = SimpleNamespace(
        user=None,
        sampling_params=SimpleNamespace(
            user=None,
            extra_args={MEMORY_USER_ID_EXTRA_ARG: "request-00042"},
        ),
        metadata=None,
    )
    assert extract_user_id(request) == "request-00042"

    unrouted = SimpleNamespace(
        user=None,
        sampling_params=SimpleNamespace(user=None, extra_args=None),
        metadata=None,
    )
    assert extract_user_id(unrouted) is None
    assert extract_user_id(unrouted, "fallback") == "fallback"


def test_warmup_routes_the_kv_warmup_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from locomo_jasper_bench.throughput.engine import warm_up

    class _RecordingLLM(_FakeWarmupLLM):
        def __init__(self) -> None:
            super().__init__()
            self.seen_params: list[object] = []

        def generate(self, prompts, sampling_params, use_tqdm=False):
            self.generate_calls += 1
            self.seen_params.append(sampling_params)

    llm = _RecordingLLM()
    base = _FakeSamplingParams()
    routed = _FakeSamplingParams(extra_args={"locomo_memory_user_id": "request-00000"})

    warm_up(
        llm,
        _prompts([4, 6]),
        base,
        batches=1,
        seed=7,
        kv_warmup=({"prompt_token_ids": [1, 2, 3]}, routed),
    )

    (params_list,) = llm.seen_params
    assert isinstance(params_list, list)
    assert len(params_list) == 3
    assert params_list[0] is base and params_list[1] is base
    assert params_list[2] is routed
