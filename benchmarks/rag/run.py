"""Entry point of one RAG benchmark stage, as the launcher's job runs it."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.common.config import load_runtime_config
from benchmarks.common.judge import format_accuracy
from benchmarks.common.logging import configure_logging
from benchmarks.rag.config import RagBenchConfig, load_rag_config

GIB = 1024**3


def main(config_path: Path, *, runtime_path: Path | None = None, stage: str = "run") -> None:
    """Execute one RAG benchmark stage using the selected JSON files."""
    runtime = load_runtime_config(runtime_path)
    config = load_rag_config(config_path, runtime=runtime, stage=stage)
    runtime.apply_environment()
    configure_logging(config.log_level)
    STAGES[stage](config)


def _run(config: RagBenchConfig) -> None:
    from benchmarks.rag.answer_run import run_answer

    summary = run_answer(config)
    print(f"wrote results to {config.run_dir}")
    _print_summary(summary)


def _judge(config: RagBenchConfig) -> None:
    from benchmarks.rag.judge_run import judge_existing_run

    summary = judge_existing_run(config)
    print(f"judged results in {config.run_dir}")
    _print_summary(summary)


def _estimate(config: RagBenchConfig) -> None:
    from benchmarks.rag.estimate import run_estimate

    estimate = run_estimate(config)
    projected = estimate["projected_kv_cache_bytes"]
    kv_cache = "unknown" if projected is None else f"{projected / GIB:.1f} GiB"
    print(
        f"dataset={estimate['dataset']} docs={estimate['doc_count']} "
        f"queries={estimate['query_count']} chunks={estimate['chunk_count']} "
        f"corpus_tokens={estimate['corpus_total_tokens']} projected_kv_cache={kv_cache}"
    )


def _preembed(config: RagBenchConfig) -> None:
    from benchmarks.rag.preembed import preembed_rag_embeddings

    summary = preembed_rag_embeddings(config)
    cache = summary["cache"]
    print(f"materialized RAG embeddings in {cache['cache_dir']}")
    print(
        f"chunks={summary['chunk_embedding_count']} queries={summary['query_embedding_count']} "
        f"cache_hits={cache['hits']} cache_misses={cache['misses']}"
    )


def _precompute_kv(config: RagBenchConfig) -> None:
    from benchmarks.rag.kv_precompute import precompute_rag_kv

    summary = precompute_rag_kv(config)
    print(
        f"rag kv chunk cache ready in {summary['cache_dir']}: "
        f"encoded={summary['encoded']} already_cached={summary['skipped']} "
        f"chunks={summary['chunk_count']} bytes={summary['cache_bytes']}"
    )


def _print_summary(summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    accuracy = metrics.get("accuracy")
    counts = f"judged={summary['judged_count']}/{summary['question_count']}"
    head = (
        f"accuracy={accuracy:.4f}"
        if accuracy is not None
        else f"questions={summary['question_count']}"
    )
    print(
        f"{head} {counts} em={format_accuracy(metrics.get('exact_match'))} "
        f"f1={format_accuracy(metrics.get('f1'))}"
    )


STAGES: dict[str, Callable[[RagBenchConfig], None]] = {
    "run": _run,
    "judge": _judge,
    "estimate": _estimate,
    "preembed": _preembed,
    "precompute-kv": _precompute_kv,
}
