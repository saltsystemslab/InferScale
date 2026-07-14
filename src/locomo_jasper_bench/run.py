from __future__ import annotations

import os
import sys

from loguru import logger

from .runtime_paths import configure_runtime_environment

configure_runtime_environment()


def main(argv: list[str] | None = None) -> None:
    from .config import parse_args
    from .runner import judge_existing_run, run_benchmark

    config = parse_args(argv)
    _configure_logging()
    if config.judge_only:
        summary = judge_existing_run(config)
        print(f"judged results in {config.run_dir}")
        accuracy = summary.get("metrics", {}).get("accuracy")
        if accuracy is not None:
            print(f"accuracy={accuracy:.4f} judged={summary['judged_count']}/{summary['question_count']}")
        else:
            print(f"questions={summary['question_count']} judged={summary['judged_count']}")
        return
    if config.check_catalogs:
        from .retrieval.memory_builder import missing_fact_catalogs

        missing = missing_fact_catalogs(config)
        if missing:
            print(
                f"missing Mem0 fact catalogs for model {config.memory_llm_model} "
                f"({len(missing)} sample(s)):",
                file=sys.stderr,
            )
            for sample_id, path in missing:
                print(f"  {sample_id}: {path}", file=sys.stderr)
            print(
                "Extraction always uses the answer model; materialize them with:\n"
                f'  EXTRACTION_MODELS="{config.model}" bash scripts/extract_facts.sh',
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"fact catalogs complete for model {config.memory_llm_model}")
        return
    if config.preembed_only:
        from .embedding.preembed import preembed_locomo_embeddings

        summary = preembed_locomo_embeddings(config)
        print(f"materialized Mem0 fact catalogs and embeddings in {summary['cache']['cache_dir']}")
        print(
            f"samples={summary['sample_count']} turns={summary['turn_embedding_count']} "
            f"inferred_memories={summary['inferred_memory_count']} "
            f"questions={summary['question_embedding_count']} cache_misses={summary['cache']['misses']} "
            f"inference_cache_misses={summary['memory_inference_cache']['misses']}"
        )
        return
    summary = run_benchmark(config)
    print(f"wrote results to {config.run_dir}")
    accuracy = summary.get("metrics", {}).get("accuracy")
    if accuracy is not None:
        print(f"accuracy={accuracy:.4f} judged={summary['judged_count']}/{summary['question_count']}")
    else:
        print(f"questions={summary['question_count']} judged={summary['judged_count']}")


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.environ.get("LOCOMO_LOG_LEVEL", "INFO").upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    )


if __name__ == "__main__":
    main()
