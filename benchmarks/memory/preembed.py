"""The preembed stage: build every sample's Mem0 fact catalog and warm the embedding caches.

Samples are processed by a thread pool because each one is bound by calls to
the extraction LLM and the embedding endpoint.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from loguru import logger

from benchmarks.common.files import write_json
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, load_locomo
from benchmarks.memory.mem0.memory_builder import SampleMemoryBuilder
from benchmarks.memory.mem0.memory_handles import memory_embedder

COUNTERS = (
    "turn_count",
    "question_count",
    "entity_embedding_count",
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
    write_json(config.run_dir / "config.json", config.to_jsonable())

    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    _require_unique_sample_ids(samples)
    worker_count = min(config.preembed_workers, len(samples)) if samples else 0
    logger.info(
        "Preembedding {} samples with {} concurrent conversation workers",
        len(samples),
        worker_count,
    )
    totals = _sum_counters(_run_sample_workers(config, samples, worker_count))
    summary = _summary(config, samples, worker_count, totals)
    write_json(config.run_dir / "preembedding.json", summary)
    logger.info(
        "Preembedded embeddings samples={} turns={} inferred_memories={} questions={} "
        "query_entities={} cache_hits={} cache_misses={} inference_cache_hits={} "
        "inference_cache_misses={}",
        len(samples),
        totals["turn_count"],
        totals["inferred_memory_count"],
        totals["question_count"],
        totals["entity_embedding_count"],
        totals["cache_hits"],
        totals["cache_misses"],
        totals["inference_cache_hits"],
        totals["inference_cache_misses"],
    )
    return summary


def preembed_questions(memory: Any, sample: ConversationSample) -> int:
    embedder = memory_embedder(memory)
    for qa in sample.qa:
        embedder.embed(qa.question, "search")
    return len(sample.qa)


def preembed_question_entities(memory: Any, sample: ConversationSample) -> int:
    """Cache the query-entity embeddings Mem0's hybrid search needs.

    Mem0's search extracts entities from the query and embeds them for entity
    boosts; a read-mode cache miss there is silently swallowed by Mem0 and the
    boost is dropped, changing the ranking. The dedup mirrors Mem0's (first 8
    entities, normalized-unique) so the cache keys match search-time lookups.
    """
    embedder = memory_embedder(memory)
    normalize = getattr(memory, "_normalize_entity_text", None)
    if not callable(normalize):
        raise RuntimeError("Mem0 memory has no _normalize_entity_text; cannot mirror entity dedup.")
    embedded_count = 0
    for qa in sample.qa:
        seen: set[str] = set()
        entity_texts: list[str] = []
        for _, entity_text in _extract_query_entities(qa.question)[:8]:
            key = normalize(entity_text)
            if key and key not in seen:
                seen.add(key)
                entity_texts.append(entity_text)
        if entity_texts:
            embedder.embed_batch(entity_texts, "search")
            embedded_count += len(entity_texts)
    return embedded_count


def _require_unique_sample_ids(samples: list[ConversationSample]) -> None:
    sample_ids = [sample.sample_id for sample in samples]
    duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate sample ids cannot be preembedded concurrently: {duplicates}")


def _run_sample_workers(
    config: MemoryRunConfig, samples: list[ConversationSample], worker_count: int
) -> list[dict[str, int]]:
    """Preembed every sample on the pool; results come back in sample order."""
    if not samples:
        return []
    results: list[dict[str, int] | None] = [None] * len(samples)
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="locomo-preembed")
    futures: dict[Future[dict[str, int]], int] = {}
    try:
        for index, sample in enumerate(samples):
            future = executor.submit(_preembed_sample, config, sample, index + 1, len(samples))
            futures[future] = index
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    completed = [result for result in results if result is not None]
    if len(completed) != len(samples):
        raise RuntimeError("Preembedding did not produce a result for every sample.")
    return completed


def _preembed_sample(
    config: MemoryRunConfig, sample: ConversationSample, index: int, count: int
) -> dict[str, int]:
    logger.info(
        "Preembedding sample {}/{} sample_id={} turns={} questions={}",
        index,
        count,
        sample.sample_id,
        len(sample.turns),
        len(sample.qa),
    )
    builder = SampleMemoryBuilder(config, embedding_cache_mode="write")
    built = builder.build_memory(sample, finalize_index=False)
    try:
        question_count = preembed_questions(built.memory, sample)
        entity_embedding_count = preembed_question_entities(built.memory, sample)
        built.log_cache_stats(sample.sample_id)
        cache_hits, cache_misses = _cache_counts(built.embedding_cache_stats())
        inference_hits, inference_misses = _cache_counts(built.llm_cache_stats())
        logger.info("Preembedded sample {}/{} sample_id={}", index, count, sample.sample_id)
        return {
            "turn_count": len(sample.turns),
            "question_count": question_count,
            "entity_embedding_count": entity_embedding_count,
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


def _sum_counters(results: list[dict[str, int]]) -> dict[str, int]:
    return {name: sum(result[name] for result in results) for name in COUNTERS}


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
        "question_embedding_count": totals["question_count"],
        "entity_embedding_count": totals["entity_embedding_count"],
        "cache": {
            "mode": "write",
            "cache_dir": str(config.embedding_cache_dir),
            "hits": totals["cache_hits"],
            "misses": totals["cache_misses"],
        },
        "memory_inference_cache": {
            "mode": "write",
            "cache_dir": str(config.memory_llm_cache_dir),
            "provider": config.memory_llm_provider,
            "model": config.memory_llm_model,
            "hits": totals["inference_cache_hits"],
            "misses": totals["inference_cache_misses"],
        },
        "config": config.to_jsonable(),
    }


def _extract_query_entities(question: str) -> list[tuple[str, str]]:
    from mem0.utils.entity_extraction import extract_entities

    return extract_entities(question)
