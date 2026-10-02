from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def main(config_path: Path, *, runtime_path: Path | None = None, stage: str = "run") -> None:
    """Execute one memory benchmark stage using the selected JSON files."""
    from benchmarks.common.config import load_runtime_config
    from benchmarks.memory.config import load_memory_config

    runtime = load_runtime_config(runtime_path)
    config = load_memory_config(config_path, runtime, stage=stage)
    runtime.apply_environment()
    _configure_logging(config.log_level)
    from benchmarks.memory.runner import judge_existing_run, run_benchmark

    if stage == "judge":
        summary = judge_existing_run(config)
        print(f"judged results in {config.run_dir}")
        accuracy = summary.get("metrics", {}).get("accuracy")
        if accuracy is not None:
            print(f"accuracy={accuracy:.4f} judged={summary['judged_count']}/{summary['question_count']}")
        else:
            print(f"questions={summary['question_count']} judged={summary['judged_count']}")
        return
    if stage == "check-catalogs":
        from benchmarks.memory.mem0.memory_builder import missing_fact_catalogs

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
                "Extraction always uses the answer model. Select the same memory JSON file in "
                "the extraction launch plan, then run bash scripts/extract_facts.sh.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"fact catalogs complete for model {config.memory_llm_model}")
        return
    if stage == "preembed":
        from benchmarks.memory.preembed import preembed_locomo_embeddings

        summary = preembed_locomo_embeddings(config)
        print(f"materialized Mem0 fact catalogs and embeddings in {summary['cache']['cache_dir']}")
        print(
            f"samples={summary['sample_count']} turns={summary['turn_embedding_count']} "
            f"inferred_memories={summary['inferred_memory_count']} "
            f"questions={summary['question_embedding_count']} cache_misses={summary['cache']['misses']} "
            f"inference_cache_misses={summary['memory_inference_cache']['misses']}"
        )
        return
    if stage == "precompute-kv":
        from benchmarks.memory.precompute_kv import precompute_kv_chunks

        summary = precompute_kv_chunks(config)
        print(
            f"kv chunk cache ready in {summary['cache_dir']}: "
            f"encoded={summary['encoded']} already_cached={summary['skipped']} "
            f"samples={summary['samples']}"
        )
        return
    summary = run_benchmark(config)
    print(f"wrote results to {config.run_dir}")
    accuracy = summary.get("metrics", {}).get("accuracy")
    if accuracy is not None:
        print(f"accuracy={accuracy:.4f} judged={summary['judged_count']}/{summary['question_count']}")
    else:
        print(f"questions={summary['question_count']} judged={summary['judged_count']}")


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}",
    )
