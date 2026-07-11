from __future__ import annotations

from pathlib import Path
import pytest

from locomo_jasper_bench.throughput.config import ThroughputConfig
from locomo_jasper_bench.throughput.reporting import RESULT_COLUMNS, validate_result_row
from locomo_jasper_bench.throughput.worker import (
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
        wall_time_s=2.0,
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
        total_input_tokens=10000,
        total_output_tokens=1000,
    )

    assert tuple(row) == RESULT_COLUMNS
    assert row["throughput_qps"] == 10.0  # 20 requests / 2s wall
    assert row["fact_count"] == 212.5
    assert row["kv_verify_time_s"] == 0.05
    assert validate_result_row(row) == validate_result_row(dict(row))


def test_run_condition_rejects_removed_prompt_injection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported condition"):
        run_condition(_config(tmp_path), "prompt_injection", (2,))


def test_run_condition_requires_single_count_kv_worker(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one user count"):
        run_condition(_config(tmp_path), "kv_injection", (2, 3))
