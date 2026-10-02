from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def main(
    config_path: Path, *, runtime_path: Path | None = None, stage: str = "run"
) -> None:
    from benchmarks.common.config import load_runtime_config
    from benchmarks.rag.config import load_rag_config

    runtime = load_runtime_config(runtime_path)
    config = load_rag_config(config_path, runtime=runtime, stage=stage)
    runtime.apply_environment()
    _configure_logging(config.log_level)
    if stage == "judge":
        from benchmarks.rag.runner import judge_existing_run

        summary = judge_existing_run(config)
        print(f"judged results in {config.run_dir}")
        _print_summary_line(summary)
        return
    if stage == "estimate":
        from benchmarks.rag.runner import run_estimate

        run_estimate(config)
        return
    if stage == "preembed":
        from benchmarks.rag.preembed import preembed_rag_embeddings

        summary = preembed_rag_embeddings(config)
        cache = summary["cache"]
        print(f"materialized RAG embeddings in {cache['cache_dir']}")
        print(
            f"chunks={summary['chunk_embedding_count']} queries={summary['query_embedding_count']} "
            f"cache_hits={cache['hits']} cache_misses={cache['misses']}"
        )
        return
    if stage == "precompute-kv":
        from benchmarks.rag.kv_precompute import precompute_rag_kv

        summary = precompute_rag_kv(config)
        print(
            f"rag kv chunk cache ready in {summary['cache_dir']}: "
            f"encoded={summary['encoded']} already_cached={summary['skipped']} "
            f"chunks={summary['chunk_count']} bytes={summary['cache_bytes']}"
        )
        return
    from benchmarks.rag.runner import run_answer

    summary = run_answer(config)
    print(f"wrote results to {config.run_dir}")
    _print_summary_line(summary)


def _print_summary_line(summary: dict) -> None:
    metrics = summary.get("metrics", {})
    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        print(
            f"accuracy={accuracy:.4f} judged={summary['judged_count']}/{summary['question_count']} "
            f"em={_fmt(metrics.get('exact_match'))} f1={_fmt(metrics.get('f1'))}"
        )
    else:
        print(
            f"questions={summary['question_count']} judged={summary['judged_count']} "
            f"em={_fmt(metrics.get('exact_match'))} f1={_fmt(metrics.get('f1'))}"
        )


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    )
