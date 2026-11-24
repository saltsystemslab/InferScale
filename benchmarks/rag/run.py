from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.config import load_runtime_config
from benchmarks.common.judge import format_accuracy
from benchmarks.common.logging import configure_logging
from benchmarks.rag.config import RagBenchConfig, load_rag_config

GIB = 1024**3


def main(config_path: Path, *, runtime_path: Path | None = None, stage: str = "run") -> None:
    runtime = load_runtime_config(runtime_path)
    config = load_rag_config(config_path, runtime=runtime, stage=stage)
    runtime.apply_environment()
    configure_logging(config.log_level)
    run_stage = STAGE_RUNNERS[stage]
    run_stage(config)


def _run(config: RagBenchConfig) -> None:
    from benchmarks.rag.answer_run import run_answer

    summary = run_answer(config)
    logger.info("wrote results to {}", config.run_dir)
    _log_summary(summary)


def _judge(config: RagBenchConfig) -> None:
    from benchmarks.rag.judge_run import judge_existing_run

    summary = judge_existing_run(config)
    logger.info("judged results in {}", config.run_dir)
    _log_summary(summary)


def _estimate(config: RagBenchConfig) -> None:
    from benchmarks.rag.estimate import run_estimate

    estimate = run_estimate(config)
    projected_bytes = estimate["projected_kv_cache_bytes"]
    projected_kv_cache = (
        "unknown" if projected_bytes is None else f"{projected_bytes / GIB:.1f} GiB"
    )
    logger.info(
        "dataset={} docs={} queries={} chunks={} corpus_tokens={} projected_kv_cache={}",
        estimate["dataset"],
        estimate["doc_count"],
        estimate["query_count"],
        estimate["chunk_count"],
        estimate["corpus_total_tokens"],
        projected_kv_cache,
    )


def _preembed(config: RagBenchConfig) -> None:
    from benchmarks.rag.preembed import preembed_rag_embeddings

    summary = preembed_rag_embeddings(config)
    embedding_cache = summary["cache"]
    logger.info("materialized RAG embeddings in {}", embedding_cache["cache_dir"])
    logger.info(
        "chunks={} cache_hits={} cache_misses={}",
        summary["chunk_embedding_count"],
        embedding_cache["hits"],
        embedding_cache["misses"],
    )


def _precompute_kv(config: RagBenchConfig) -> None:
    from benchmarks.rag.kv_precompute import precompute_rag_kv

    summary = precompute_rag_kv(config)
    logger.info(
        "rag kv chunk cache ready in {}: encoded={} already_cached={} chunks={} bytes={}",
        summary["cache_dir"],
        summary["encoded"],
        summary["skipped"],
        summary["chunk_count"],
        summary["cache_bytes"],
    )


def _log_summary(summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    accuracy = metrics.get("accuracy")
    judged = f"judged={summary['judged_count']}/{summary['question_count']}"
    if accuracy is not None:
        headline = f"accuracy={accuracy:.4f}"
    else:
        headline = f"questions={summary['question_count']}"
    logger.info(
        "{} {} em={} f1={}",
        headline,
        judged,
        format_accuracy(metrics.get("exact_match")),
        format_accuracy(metrics.get("f1")),
    )


STAGE_RUNNERS: dict[str, Callable[[RagBenchConfig], None]] = {
    "run": _run,
    "judge": _judge,
    "estimate": _estimate,
    "preembed": _preembed,
    "precompute-kv": _precompute_kv,
}
