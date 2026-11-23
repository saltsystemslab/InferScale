from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.config import RuntimeConfig, load_runtime_config, stamp_now
from benchmarks.common.files import read_jsonl
from benchmarks.common.json_fields import expand_path, load_json_object
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.sweeps import MemoryCell

DEFAULT_CONFIG = Path("configs/memory/accuracy-latency/llama.json")
FACTS_PER_QUESTION = 100
PRECEDING_TURNS = 50
WARMUP = 3
SIDES: dict[str, MemoryCell] = {
    "mem0": MemoryCell(
        variant="prompt-injection",
        answer_backend="prompt-injection",
        vector_backend="qdrant",
        top_k=FACTS_PER_QUESTION,
        context_window=0,
    ),
    "lmcache": MemoryCell(
        variant="lmcache-blend",
        answer_backend="lmcache-blend",
        vector_backend="qdrant",
        top_k=FACTS_PER_QUESTION,
        context_window=0,
    ),
    "inferscale": MemoryCell(
        variant="kv",
        answer_backend="kv-injection",
        vector_backend="jasper",
        top_k=FACTS_PER_QUESTION,
        context_window=PRECEDING_TURNS,
    ),
}


def measure_side(mode: str, *, config_path: Path, questions: int, out: Path) -> Path:
    runtime = load_runtime_config()
    config = race_config(mode, runtime, config_path=config_path, questions=questions)
    out.parent.mkdir(parents=True, exist_ok=True)
    runtime.apply_environment()

    from benchmarks.memory.prediction import PREDICTIONS_FILE
    from benchmarks.memory.runner import run_benchmark

    run_benchmark(config)
    rows = timed_rows(config.run_dir / PREDICTIONS_FILE)
    for index, row in enumerate(rows):
        logger.info("{:>2}  TTFT {:>7.1f} ms  {}", index, row["ttft_ms"], row["question"])
    timeline = {
        "mode": mode,
        "model": config.model,
        "top_k": config.top_k,
        "context_window": config.context_window,
        "rows": rows,
    }
    out.write_text(json.dumps(timeline, indent=1))
    logger.info("wrote {}; the full benchmark run is in {}", out, config.run_dir)
    return config.run_dir


def race_config(
    mode: str, runtime: RuntimeConfig, *, config_path: Path, questions: int
) -> MemoryRunConfig:
    cell = SIDES[mode]
    authored = load_json_object(expand_path(config_path, root=runtime.root))
    run = cell.materialize(authored)
    run.update(
        run_id=f"ttft-race-{mode}-{stamp_now()}",
        max_samples=1,
        max_questions=WARMUP + questions,
        skip_judge=True,
    )
    return MemoryRunConfig.from_dict(run, runtime=runtime)


def timed_rows(predictions_path: Path) -> list[dict[str, Any]]:
    timed_records = read_jsonl(predictions_path)[WARMUP:]
    return [
        {
            "question": record["question"],
            "gold": record["gold_answer"],
            "answer": record["predicted_answer"],
            "ttft_ms": record["metrics"]["time_to_first_token_ms"],
            "total_ms": record["metrics"]["answer_generate_time_ms"],
        }
        for record in timed_records
    ]
