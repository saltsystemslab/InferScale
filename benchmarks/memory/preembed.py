from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from loguru import logger

from benchmarks.common.files import write_json
from benchmarks.common.paths import RUN_CONFIG_FILE
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, load_locomo
from benchmarks.memory.mem0.memory_builder import SampleMemoryBuilder

PREEMBEDDING_FILE = "preembedding.json"
CACHE_MODE = "write"
COUNTERS = (
    "turn_count",
    "cache_hits",
    "cache_misses",
    "inferred_memory_count",
    "inference_cache_hits",
    "inference_cache_misses",
)


def preembed_locomo_embeddings(config: MemoryRunConfig) -> dict[str, Any]:
    if not config.embedding_cache_enabled:
        raise RuntimeError("The preembed stage requires embedding_cache_enabled=true in JSON.")
    if config.preembed_workers < 1:
        raise ValueError("preembed_workers must be >= 1.")
    logger.info(
        "Materializing Mem0 fact catalogs and embeddings dataset={} cache_dir={}",
        config.dataset_path,
        config.embedding_cache_dir,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / RUN_CONFIG_FILE, config.to_jsonable())

    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    _require_unique_sample_ids(samples)
    worker_count = min(config.preembed_workers, len(samples))
    logger.info(
        "Preembedding {} samples with {} concurrent conversation workers",
        len(samples),
        worker_count,
    )
    totals = _sum_counters(_run_sample_workers(config, samples, worker_count))
    summary = _summary(config, samples, worker_count, totals)
    write_json(config.run_dir / PREEMBEDDING_FILE, summary)
    logger.info(
        "Preembedded embeddings samples={} turns={} inferred_memories={} "
        "cache_hits={} cache_misses={} inference_cache_hits={} inference_cache_misses={}",
        len(samples),
        totals["turn_count"],
        totals["inferred_memory_count"],
        totals["cache_hits"],
        totals["cache_misses"],
        totals["inference_cache_hits"],
        totals["inference_cache_misses"],
    )
    return summary


def _require_unique_sample_ids(samples: list[ConversationSample]) -> None:
    sample_ids = [sample.sample_id for sample in samples]
    duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate sample ids cannot be preembedded concurrently: {duplicates}")


def _run_sample_workers(
    config: MemoryRunConfig, samples: list[ConversationSample], worker_count: int
) -> list[dict[str, int]]:
    if not samples:
        return []
    sample_counters: list[dict[str, int] | None] = [None] * len(samples)
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="locomo-preembed")
    sample_index_of: dict[Future[dict[str, int]], int] = {}
    try:
        for sample_index, sample in enumerate(samples):
            future = executor.submit(
                _preembed_sample, config, sample, sample_index + 1, len(samples)
            )
            sample_index_of[future] = sample_index
        for future in as_completed(sample_index_of):
            sample_counters[sample_index_of[future]] = future.result()
    except BaseException:
        for future in sample_index_of:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    completed = [counters for counters in sample_counters if counters is not None]
    if len(completed) != len(samples):
        raise RuntimeError("Preembedding did not produce a result for every sample.")
    return completed


def _preembed_sample(
    config: MemoryRunConfig, sample: ConversationSample, sample_number: int, sample_count: int
) -> dict[str, int]:
    logger.info(
        "Preembedding sample {}/{} sample_id={} turns={}",
        sample_number,
        sample_count,
        sample.sample_id,
        len(sample.turns),
    )
    memory_builder = SampleMemoryBuilder(config, embedding_cache_mode=CACHE_MODE)
    built = memory_builder.build_memory(sample, finalize_index=False)
    try:
        built.log_cache_stats(sample.sample_id)
        cache_hits, cache_misses = _cache_counts(built.embedding_cache_stats())
        inference_hits, inference_misses = _cache_counts(built.llm_cache_stats())
        logger.info(
            "Preembedded sample {}/{} sample_id={}", sample_number, sample_count, sample.sample_id
        )
        return {
            "turn_count": len(sample.turns),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "inferred_memory_count": int(built.metrics.get("memory_inferred_record_count") or 0),
            "inference_cache_hits": inference_hits,
            "inference_cache_misses": inference_misses,
        }
    finally:
        built.close()


def _cache_counts(stats: dict[str, Any] | None) -> tuple[int, int]:
    if stats is None:
        return 0, 0
    return int(stats.get("hits", 0)), int(stats.get("misses", 0))


def _sum_counters(sample_counters: list[dict[str, int]]) -> dict[str, int]:
    return {name: sum(counters[name] for counters in sample_counters) for name in COUNTERS}


def _summary(
    config: MemoryRunConfig,
    samples: list[ConversationSample],
    worker_count: int,
    totals: dict[str, int],
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "mode": "preembed",
        "sample_count": len(samples),
        "preembed_workers": worker_count,
        "turn_embedding_count": totals["turn_count"],
        "inferred_memory_count": totals["inferred_memory_count"],
        "cache": {
            "mode": CACHE_MODE,
            "cache_dir": str(config.embedding_cache_dir),
            "hits": totals["cache_hits"],
            "misses": totals["cache_misses"],
        },
        "memory_inference_cache": {
            "mode": CACHE_MODE,
            "cache_dir": str(config.memory_llm_cache_dir),
            "provider": config.memory_llm_provider,
            "model": config.memory_llm_model,
            "hits": totals["inference_cache_hits"],
            "misses": totals["inference_cache_misses"],
        },
        "config": config.to_jsonable(),
    }
