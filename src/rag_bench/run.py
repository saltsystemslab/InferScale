from __future__ import annotations

import os
import sys

from loguru import logger

from locomo_jasper_bench.runtime_paths import configure_runtime_environment

configure_runtime_environment()


def main(argv: list[str] | None = None) -> None:
    from .config import parse_args

    config = parse_args(argv)
    _configure_logging()
    if config.judge_only:
        from .runner import judge_existing_run

        summary = judge_existing_run(config)
        print(f"judged results in {config.run_dir}")
        _print_summary_line(summary)
        return
    if config.estimate_only:
        from .runner import run_estimate

        run_estimate(config)
        return
    if config.preembed_only:
        from .preembed import preembed_rag_embeddings

        summary = preembed_rag_embeddings(config)
        cache = summary["cache"]
        print(f"materialized RAG embeddings in {cache['cache_dir']}")
        print(
            f"chunks={summary['chunk_embedding_count']} queries={summary['query_embedding_count']} "
            f"cache_hits={cache['hits']} cache_misses={cache['misses']}"
        )
        return
    if config.precompute_kv_only:
        from .kv_precompute import precompute_rag_kv

        summary = precompute_rag_kv(config)
        print(
            f"rag kv chunk cache ready in {summary['cache_dir']}: "
            f"encoded={summary['encoded']} already_cached={summary['skipped']} "
            f"chunks={summary['chunk_count']} bytes={summary['cache_bytes']}"
        )
        return
    from .runner import run_answer

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


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=(os.environ.get("RAG_LOG_LEVEL") or os.environ.get("LOCOMO_LOG_LEVEL") or "INFO").upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    )


if __name__ == "__main__":
    main()
