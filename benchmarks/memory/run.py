"""Entry point of one memory benchmark stage, as the launcher's job runs it."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.common.config import load_runtime_config
from benchmarks.common.logging import configure_logging
from benchmarks.memory.config import MemoryRunConfig, load_memory_config


def main(config_path: Path, *, runtime_path: Path | None = None, stage: str = "run") -> None:
    """Execute one memory benchmark stage using the selected JSON files."""
    runtime = load_runtime_config(runtime_path)
    config = load_memory_config(config_path, runtime, stage=stage)
    # The stage modules import Mem0, which reads the runtime's environment
    # (MEM0_DIR among others) when first imported, so it is applied first.
    runtime.apply_environment()
    configure_logging(config.log_level)
    STAGES[stage](config)


def _run(config: MemoryRunConfig) -> None:
    from benchmarks.memory.runner import run_benchmark

    summary = run_benchmark(config)
    print(f"wrote results to {config.run_dir}")
    _print_accuracy(summary)


def _judge(config: MemoryRunConfig) -> None:
    from benchmarks.memory.runner import judge_existing_run

    summary = judge_existing_run(config)
    print(f"judged results in {config.run_dir}")
    _print_accuracy(summary)


def _check_catalogs(config: MemoryRunConfig) -> None:
    from benchmarks.memory.mem0.memory_builder import missing_fact_catalogs

    missing = missing_fact_catalogs(config)
    if not missing:
        print(f"fact catalogs complete for model {config.memory_llm_model}")
        return
    print(
        f"missing Mem0 fact catalogs for model {config.memory_llm_model} "
        f"({len(missing)} sample(s)):",
        file=sys.stderr,
    )
    for sample_id, path in missing:
        print(f"  {sample_id}: {path}", file=sys.stderr)
    print(
        "Extraction always uses the answer model. Select the same memory JSON file in "
        "the extraction launch plan, then run bash scripts/extract_facts.sh.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _preembed(config: MemoryRunConfig) -> None:
    from benchmarks.memory.preembed import preembed_locomo_embeddings

    summary = preembed_locomo_embeddings(config)
    print(f"materialized Mem0 fact catalogs and embeddings in {summary['cache']['cache_dir']}")
    print(
        f"samples={summary['sample_count']} turns={summary['turn_embedding_count']} "
        f"inferred_memories={summary['inferred_memory_count']} "
        f"questions={summary['question_embedding_count']} "
        f"cache_misses={summary['cache']['misses']} "
        f"inference_cache_misses={summary['memory_inference_cache']['misses']}"
    )


def _precompute_kv(config: MemoryRunConfig) -> None:
    from benchmarks.memory.precompute_kv import precompute_kv_chunks

    summary = precompute_kv_chunks(config)
    print(
        f"kv chunk cache ready in {summary['cache_dir']}: "
        f"encoded={summary['encoded']} already_cached={summary['skipped']} "
        f"samples={summary['samples']}"
    )


def _print_accuracy(summary: dict[str, Any]) -> None:
    accuracy = summary.get("metrics", {}).get("accuracy")
    judged = f"judged={summary['judged_count']}/{summary['question_count']}"
    if accuracy is not None:
        print(f"accuracy={accuracy:.4f} {judged}")
    else:
        print(f"questions={summary['question_count']} {judged}")


STAGES: dict[str, Callable[[MemoryRunConfig], None]] = {
    "run": _run,
    "judge": _judge,
    "check-catalogs": _check_catalogs,
    "preembed": _preembed,
    "precompute-kv": _precompute_kv,
}
