from __future__ import annotations

import pytest

from locomo_jasper_bench.config import BenchmarkConfig
from locomo_jasper_bench.modes import existing_run_mode, result_mode


@pytest.mark.parametrize(
    ("answer_backend", "expected"),
    [
        ("vllm-kv", "mem0-kv"),
        ("vllm-prefix", "mem0-prefix"),
    ],
)
def test_result_mode_names_mem0_memory_unit(answer_backend: str, expected: str) -> None:
    config = BenchmarkConfig(answer_backend=answer_backend)  # type: ignore[arg-type]

    assert result_mode(config) == expected


def test_existing_run_mode_uses_mem0_marker_for_new_runs() -> None:
    config = BenchmarkConfig(answer_backend="vllm-prefix")

    assert existing_run_mode(
        {"answer_backend": "vllm-kv", "memory_unit": "mem0-fact"},
        [{"mode": "vllm-kv"}],
        config,
    ) == "mem0-kv"


def test_existing_run_mode_preserves_legacy_record_mode() -> None:
    config = BenchmarkConfig(answer_backend="vllm-prefix")

    assert existing_run_mode(
        {},
        [{"mode": "legacy-kv"}],
        config,
    ) == "legacy-kv"
