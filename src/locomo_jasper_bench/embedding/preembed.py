from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any

from loguru import logger

from ..config import BenchmarkConfig
from ..data import ConversationSample, load_locomo
from ..retrieval.memory_builder import SampleMemoryBuilder, memory_embedder
from ..results import write_json


def preembed_locomo_embeddings(config: BenchmarkConfig) -> dict[str, Any]:
    if not config.embedding_cache_enabled:
        raise RuntimeError("--preembed-only requires the embedding cache; remove --no-embedding-cache.")
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
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = sorted({sample_id for sample_id in sample_ids if sample_ids.count(sample_id) > 1})
        raise ValueError(f"Duplicate sample ids cannot be preembedded concurrently: {duplicates}")

    worker_count = min(config.preembed_workers, len(samples)) if samples else 0
    logger.info(
        "Preembedding {} samples with {} concurrent conversation workers",
        len(samples),
        worker_count,
    )
    sample_results: list[dict[str, int] | None] = [None] * len(samples)
    if samples:
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="locomo-preembed",
        )
        futures: dict[Future[dict[str, int]], int] = {}
        try:
            for sample_index, sample in enumerate(samples):
                future = executor.submit(
                    _preembed_sample,
                    config,
                    sample,
                    sample_index + 1,
                    len(samples),
                )
                futures[future] = sample_index
            for future in as_completed(futures):
                sample_index = futures[future]
                sample_results[sample_index] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    completed_results = [result for result in sample_results if result is not None]
    if len(completed_results) != len(samples):
        raise RuntimeError("Preembedding did not produce a result for every sample.")
    turn_count = sum(result["turn_count"] for result in completed_results)
    question_count = sum(result["question_count"] for result in completed_results)
    entity_embedding_count = sum(result["entity_embedding_count"] for result in completed_results)
    cache_hits = sum(result["cache_hits"] for result in completed_results)
    cache_misses = sum(result["cache_misses"] for result in completed_results)
    inferred_memory_count = sum(result["inferred_memory_count"] for result in completed_results)
    inference_cache_hits = sum(result["inference_cache_hits"] for result in completed_results)
    inference_cache_misses = sum(result["inference_cache_misses"] for result in completed_results)

    summary = {
        "run_id": config.run_id,
        "mode": "preembed",
        "sample_count": len(samples),
        "preembed_workers": worker_count,
        "turn_embedding_count": turn_count,
        "inferred_memory_count": inferred_memory_count,
        "question_embedding_count": question_count,
        "entity_embedding_count": entity_embedding_count,
        "cache": {
            "mode": "write",
            "cache_dir": str(config.embedding_cache_dir),
            "hits": cache_hits,
            "misses": cache_misses,
        },
        "memory_inference_cache": {
            "mode": "write",
            "cache_dir": str(config.memory_llm_cache_dir),
            "provider": config.memory_llm_provider,
            "model": config.memory_llm_model,
            "hits": inference_cache_hits,
            "misses": inference_cache_misses,
        },
        "config": config.to_jsonable(),
    }
    write_json(config.run_dir / "preembedding.json", summary)
    logger.info(
        "Preembedded embeddings samples={} turns={} inferred_memories={} questions={} "
        "query_entities={} cache_hits={} cache_misses={} inference_cache_hits={} "
        "inference_cache_misses={}",
        len(samples),
        turn_count,
        inferred_memory_count,
        question_count,
        entity_embedding_count,
        cache_hits,
        cache_misses,
        inference_cache_hits,
        inference_cache_misses,
    )
    return summary


def _preembed_sample(
    config: BenchmarkConfig,
    sample: ConversationSample,
    sample_index: int,
    sample_count: int,
) -> dict[str, int]:
    logger.info(
        "Preembedding sample {}/{} sample_id={} turns={} questions={}",
        sample_index,
        sample_count,
        sample.sample_id,
        len(sample.turns),
        len(sample.qa),
    )
    memory_builder = SampleMemoryBuilder(config, embedding_cache_mode="write")
    memory, memory_metrics = memory_builder.build_with_metrics(sample, finalize_index=False)
    try:
        question_count = preembed_questions(memory, sample)
        entity_embedding_count = preembed_question_entities(memory, sample)
        cache_hits = 0
        cache_misses = 0
        stats = memory_builder.embedding_cache_stats(memory)
        if stats is not None:
            cache_hits = int(stats["hits"])
            cache_misses = int(stats["misses"])
            memory_builder.log_embedding_cache_stats(memory, sample.sample_id)
        inference_cache_hits = 0
        inference_cache_misses = 0
        inference_stats = memory_builder.memory_llm_cache_stats(memory)
        if inference_stats is not None:
            inference_cache_hits = int(inference_stats["hits"])
            inference_cache_misses = int(inference_stats["misses"])
        logger.info("Preembedded sample {}/{} sample_id={}", sample_index, sample_count, sample.sample_id)
        return {
            "turn_count": len(sample.turns),
            "question_count": question_count,
            "entity_embedding_count": entity_embedding_count,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "inferred_memory_count": int(memory_metrics.get("memory_inferred_record_count") or 0),
            "inference_cache_hits": inference_cache_hits,
            "inference_cache_misses": inference_cache_misses,
        }
    finally:
        memory_builder.close(memory)


def preembed_questions(memory: Any, sample: ConversationSample) -> int:
    embedder = memory_embedder(memory)
    for qa in sample.qa:
        embedder.embed(qa.question, "search")
    return len(sample.qa)


def preembed_question_entities(memory: Any, sample: ConversationSample) -> int:
    """Cache the query-entity embeddings mem0's hybrid search needs.

    Mem0's search extracts entities from the query and embeds them for entity
    boosts; a read-mode cache miss there is silently swallowed by mem0 and the
    boost is dropped, changing ranking. Mirror mem0's dedup (first 8 entities,
    normalized-unique) so the cache keys match search-time lookups exactly.
    """
    embedder = memory_embedder(memory)
    normalize = getattr(memory, "_normalize_entity_text", None)
    if not callable(normalize):
        raise RuntimeError(
            "Mem0 memory has no _normalize_entity_text; cannot mirror entity dedup."
        )
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


def _extract_query_entities(question: str) -> list[tuple[str, str]]:
    from mem0.utils.entity_extraction import extract_entities

    return extract_entities(question)
