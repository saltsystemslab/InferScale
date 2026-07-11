from __future__ import annotations

from typing import Any

from loguru import logger

from ..config import BenchmarkConfig
from ..data import ConversationSample, load_locomo
from ..retrieval.memory_builder import SampleMemoryBuilder, memory_embedder
from ..results import write_json


def preembed_locomo_embeddings(config: BenchmarkConfig) -> dict[str, Any]:
    if not config.embedding_cache_enabled:
        raise RuntimeError("--preembed-only requires the embedding cache; remove --no-embedding-cache.")

    logger.info(
        "Materializing Mem0 fact catalogs and embeddings dataset={} cache_dir={}",
        config.dataset_path,
        config.embedding_cache_dir,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.json", config.to_jsonable())

    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    memory_builder = SampleMemoryBuilder(config, embedding_cache_mode="write")
    turn_count = 0
    question_count = 0
    entity_embedding_count = 0
    cache_hits = 0
    cache_misses = 0
    inferred_memory_count = 0
    inference_cache_hits = 0
    inference_cache_misses = 0

    for sample_index, sample in enumerate(samples, start=1):
        logger.info(
            "Preembedding sample {}/{} sample_id={} turns={} questions={}",
            sample_index,
            len(samples),
            sample.sample_id,
            len(sample.turns),
            len(sample.qa),
        )
        memory, memory_metrics = memory_builder.build_with_metrics(sample, finalize_index=False)
        try:
            turn_count += len(sample.turns)
            inferred_memory_count += int(memory_metrics.get("memory_inferred_record_count") or 0)
            question_count += preembed_questions(memory, sample)
            entity_embedding_count += preembed_question_entities(memory, sample)
            stats = memory_builder.embedding_cache_stats(memory)
            if stats is not None:
                cache_hits += int(stats["hits"])
                cache_misses += int(stats["misses"])
                memory_builder.log_embedding_cache_stats(memory, sample.sample_id)
            inference_stats = memory_builder.memory_llm_cache_stats(memory)
            if inference_stats is not None:
                inference_cache_hits += int(inference_stats["hits"])
                inference_cache_misses += int(inference_stats["misses"])
        finally:
            memory_builder.close(memory)

    summary = {
        "run_id": config.run_id,
        "mode": "preembed",
        "sample_count": len(samples),
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
