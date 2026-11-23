from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.common.config import RuntimeConfig, load_runtime_config, stamp_now
from benchmarks.common.files import read_jsonl
from benchmarks.common.json_fields import expand_path, load_json_object
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.sweeps import MemoryCell

DEFAULT_CONFIG = Path("configs/memory/accuracy-latency/llama.json")
TOP_K = 100  # facts retrieved per question, on both sides
CONTEXT_WINDOW = 50  # preceding turns in each fact's InferScale encoding prefix
WARMUP = 3  # leading questions answered before the timed ones
# The sweep cell each side runs: the memory benchmark's two baselines and its InferScale cell.
SIDES = {
    "mem0": MemoryCell(
        variant="prompt-injection",
        answer_backend="prompt-injection",
        vector_backend="qdrant",
        top_k=TOP_K,
        context_window=0,
    ),
    "lmcache": MemoryCell(
        variant="lmcache-blend",
        answer_backend="lmcache-blend",
        vector_backend="qdrant",
        top_k=TOP_K,
        context_window=0,
    ),
    "inferscale": MemoryCell(
        variant="kv",
        answer_backend="kv-injection",
        vector_backend="jasper",
        top_k=TOP_K,
        context_window=CONTEXT_WINDOW,
    ),
}


def measure_side(mode: str, *, config_path: Path, questions: int, out: Path) -> Path:
    runtime = load_runtime_config()
    config = race_config(mode, runtime, config_path=config_path, questions=questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    runtime.apply_environment()  # Cache locations, before the GPU libraries load.

    from benchmarks.memory.prediction import PREDICTIONS_FILE
    from benchmarks.memory.runner import run_benchmark

    run_benchmark(config)
    rows = timed_rows(config.run_dir / PREDICTIONS_FILE)
    for index, row in enumerate(rows):
        print(f"{index:>2}  TTFT {row['ttft_ms']:>7.1f} ms  {row['question']}")
    timeline = {
        "mode": mode,
        "model": config.model,
        "top_k": config.top_k,
        "context_window": config.context_window,
        "rows": rows,
    }
    out.write_text(json.dumps(timeline, indent=1))
    print(f"wrote {out}; the full benchmark run is in {config.run_dir}")
    return config.run_dir


def race_config(
    mode: str, runtime: RuntimeConfig, *, config_path: Path, questions: int
) -> MemoryRunConfig:
    data = SIDES[mode].materialize(load_json_object(expand_path(config_path, root=runtime.root)))
    data.update(
        run_id=f"ttft-race-{mode}-{stamp_now()}",
        max_samples=1,
        max_questions=WARMUP + questions,
        skip_judge=True,
    )
    return MemoryRunConfig.from_dict(data, runtime=runtime)


def timed_rows(predictions_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "question": record["question"],
            "gold": record["gold_answer"],
            "answer": record["predicted_answer"],
            "ttft_ms": record["metrics"]["time_to_first_token_ms"],
            "total_ms": record["metrics"]["answer_generate_time_ms"],
        }
        for record in read_jsonl(predictions_path)[WARMUP:]
    ]
