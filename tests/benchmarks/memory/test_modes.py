from __future__ import annotations

import pytest

from memory_config import make_memory_config
from benchmarks.memory.modes import result_mode


@pytest.mark.parametrize(
    ("answer_backend", "expected"),
    [
        ("kv-injection", "mem0-kv"),
        ("prompt-injection", "mem0-prompt-injection"),
    ],
)
def test_result_mode_names_mem0_memory_unit(answer_backend: str, expected: str) -> None:
    config = make_memory_config(answer_backend=answer_backend)  # type: ignore[arg-type]

    assert result_mode(config) == expected
