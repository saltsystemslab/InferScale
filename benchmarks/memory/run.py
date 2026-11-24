from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.config import load_runtime_config
from benchmarks.common.logging import configure_logging
from benchmarks.memory.config import MemoryRunConfig, load_memory_config


def main(config_path: Path, *, runtime_path: Path | None = None, stage: str = "run") -> None:
    runtime = load_runtime_config(runtime_path)
    config = load_memory_config(config_path, runtime, stage=stage)
    runtime.apply_environment()
    configure_logging(config.log_level)
    STAGES[stage](config)


def _run(config: MemoryRunConfig) -> None:
    from benchmarks.memory.runner import run_benchmark

    summary = run_benchmark(config)
    logger.info("wrote results to {}", config.run_dir)
    _log_accuracy(summary)


def _judge(config: MemoryRunConfig) -> None:
    from benchmarks.memory.runner import judge_existing_run

    summary = judge_existing_run(config)
    logger.info("judged results in {}", config.run_dir)
    _log_accuracy(summary)


def _check_catalogs(config: MemoryRunConfig) -> None:
    from benchmarks.memory.mem0.memory_builder import missing_fact_catalogs

    missing_catalogs = missing_fact_catalogs(config)
    if not missing_catalogs:
        logger.info("fact catalogs complete for model {}", config.memory_llm_model)
        return
    logger.error(
        "missing Mem0 fact catalogs for model {} ({} sample(s)):",
        config.memory_llm_model,
        len(missing_catalogs),
    )
    for sample_id, catalog_path in missing_catalogs:
        logger.error("  {}: {}", sample_id, catalog_path)
    logger.error(
        "Extraction always uses the answer model. Select the same memory JSON file in "
        "the extraction launch plan, then run bash scripts/extract_facts.sh."
    )
    raise SystemExit(1)


def _preembed(config: MemoryRunConfig) -> None:
    from benchmarks.memory.preembed import preembed_locomo_embeddings

    summary = preembed_locomo_embeddings(config)
    logger.info(
        "materialized Mem0 fact catalogs and embeddings in {}", summary["cache"]["cache_dir"]
    )
    logger.info(
        "samples={} turns={} inferred_memories={} cache_misses={} inference_cache_misses={}",
        summary["sample_count"],
        summary["turn_embedding_count"],
        summary["inferred_memory_count"],
        summary["cache"]["misses"],
        summary["memory_inference_cache"]["misses"],
    )


def _precompute_kv(config: MemoryRunConfig) -> None:
    from benchmarks.memory.precompute_kv import precompute_kv_chunks

    summary = precompute_kv_chunks(config)
    logger.info(
        "kv chunk cache ready in {}: encoded={} already_cached={} samples={}",
        summary["cache_dir"],
        summary["encoded"],
        summary["skipped"],
        summary["samples"],
    )


def _log_accuracy(summary: dict[str, Any]) -> None:
    accuracy = summary.get("metrics", {}).get("accuracy")
    judged_share = f"judged={summary['judged_count']}/{summary['question_count']}"
    if accuracy is not None:
        logger.info("accuracy={:.4f} {}", accuracy, judged_share)
    else:
        logger.info("questions={} {}", summary["question_count"], judged_share)


STAGES: dict[str, Callable[[MemoryRunConfig], None]] = {
    "run": _run,
    "judge": _judge,
    "check-catalogs": _check_catalogs,
    "preembed": _preembed,
    "precompute-kv": _precompute_kv,
}
