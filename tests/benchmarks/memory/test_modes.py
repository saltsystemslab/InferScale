from __future__ import annotations

import pytest

from memory_config import make_memory_config
from benchmarks.memory.modes import existing_run_mode, result_mode


@pytest.mark.parametrize(
    ("answer_backend", "expected"),
    [
        ("kv-injection", "mem0-kv"),
        ("prompt-injection", "mem0-prefix"),
    ],
)
def test_result_mode_names_mem0_memory_unit(answer_backend: str, expected: str) -> None:
    config = make_memory_config(answer_backend=answer_backend)  # type: ignore[arg-type]

    assert result_mode(config) == expected


def test_existing_run_mode_uses_mem0_marker_for_new_runs() -> None:
    config = make_memory_config(answer_backend="prompt-injection")

    assert existing_run_mode(
        {"answer_backend": "kv-injection", "memory_unit": "mem0-fact"},
        [{"mode": "kv-injection"}],
        config,
    ) == "mem0-kv"


def test_existing_run_mode_preserves_legacy_record_mode() -> None:
    config = make_memory_config(answer_backend="prompt-injection")

    assert existing_run_mode(
        {},
        [{"mode": "legacy-kv"}],
        config,
    ) == "legacy-kv"


@pytest.mark.parametrize(
    ("saved_backend", "expected"),
    [("vllm-kv", "mem0-kv"), ("vllm-prefix", "mem0-prefix")],
)
def test_existing_run_mode_preserves_historical_backend_names(saved_backend, expected) -> None:
    config = make_memory_config(answer_backend="prompt-injection")

    assert existing_run_mode(
        {"answer_backend": saved_backend, "memory_unit": "mem0-fact"}, [], config,
    ) == expected
