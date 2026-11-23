from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.memory.sweeps import MemoryCell
from benchmarks.ttft_race import measure

ROOT = Path(__file__).resolve().parents[3]
MODULES_LOADED_WHEN_THE_ENVIRONMENT_IS_APPLIED = """
import json
import sys
from pathlib import Path

from benchmarks.common.config import RuntimeConfig
from benchmarks.ttft_race import measure


class EnvironmentApplied(Exception):
    pass


def apply_environment(runtime):
    print(json.dumps(sorted(sys.modules)))
    raise EnvironmentApplied


RuntimeConfig.apply_environment = apply_environment
try:
    measure.measure_side(
        "inferscale", config_path=measure.DEFAULT_CONFIG, questions=1, out=Path(sys.argv[1])
    )
except EnvironmentApplied:
    pass
"""
GPU_LIBRARIES = {"torch", "vllm", "transformers", "mem0", "lmcache"}


def modules_loaded_by(script: str, *arguments: str) -> set[str]:
    process = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])},
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr
    return set(json.loads(process.stdout))


def test_sides_are_the_two_baseline_cells_and_the_inferscale_cell_of_the_memory_benchmark() -> None:
    assert list(measure.SIDES.items()) == [
        (
            "mem0",
            MemoryCell(
                variant="prompt-injection",
                answer_backend="prompt-injection",
                vector_backend="qdrant",
                top_k=100,
                context_window=0,
            ),
        ),
        (
            "lmcache",
            MemoryCell(
                variant="lmcache-blend",
                answer_backend="lmcache-blend",
                vector_backend="qdrant",
                top_k=100,
                context_window=0,
            ),
        ),
        (
            "inferscale",
            MemoryCell(
                variant="kv",
                answer_backend="kv-injection",
                vector_backend="jasper",
                top_k=100,
                context_window=50,
            ),
        ),
    ]
    assert str(measure.DEFAULT_CONFIG) == "configs/memory/accuracy-latency/llama.json"
    assert measure.WARMUP == 3


def test_cache_locations_are_applied_before_any_gpu_library_loads(tmp_path: Path) -> None:
    loaded = modules_loaded_by(
        MODULES_LOADED_WHEN_THE_ENVIRONMENT_IS_APPLIED, str(tmp_path / "inferscale.json")
    )

    assert "benchmarks.ttft_race.measure" in loaded
    assert not loaded & GPU_LIBRARIES
    assert not loaded & {"benchmarks.memory.runner", "benchmarks.memory.prediction"}


def prediction(index: int) -> dict[str, Any]:
    return {
        "question": f"Question {index}?",
        "gold_answer": f"gold {index}",
        "predicted_answer": f"answer {index}",
        "metrics": {
            "time_to_first_token_ms": 10.5 + index,
            "answer_generate_time_ms": 100.25 + index,
            "retrieval_time_ms": 1.0,
        },
        "judge": None,
    }


@pytest.mark.parametrize("record_count,timed", [(3, []), (6, [3, 4, 5])])
def test_timed_rows_leave_out_the_warmup_questions(
    tmp_path: Path, record_count: int, timed: list[int]
) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(prediction(index)) + "\n" for index in range(record_count))
    )

    rows = measure.timed_rows(predictions_path)

    assert rows == [
        {
            "question": f"Question {index}?",
            "gold": f"gold {index}",
            "answer": f"answer {index}",
            "ttft_ms": 10.5 + index,
            "total_ms": 100.25 + index,
        }
        for index in timed
    ]
    assert all(list(row) == ["question", "gold", "answer", "ttft_ms", "total_ms"] for row in rows)
