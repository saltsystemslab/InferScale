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

    logger.info("Preembedding LoCoMo embeddings dataset={} cache_dir={}", config.dataset_path, config.embedding_cache_dir)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.json", config.to_jsonable())

    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    memory_builder = SampleMemoryBuilder(config, embedding_cache_mode="write")
    session_count = 0
    turn_count = 0
    question_count = 0
    cache_hits = 0
    cache_misses = 0

    for sample_index, sample in enumerate(samples, start=1):
        logger.info(
            "Preembedding sample {}/{} sample_id={} sessions={} turns={} questions={}",
            sample_index,
            len(samples),
            sample.sample_id,
            len(sample.sessions),
            len(sample.turns),
            len(sample.qa),
        )
        memory = memory_builder.build(sample, finalize_index=False)
        try:
            session_count += len(sample.sessions)
            turn_count += len(sample.turns)
            question_count += preembed_questions(memory, sample)
            stats = memory_builder.embedding_cache_stats(memory)
            if stats is not None:
                cache_hits += int(stats["hits"])
                cache_misses += int(stats["misses"])
                memory_builder.log_embedding_cache_stats(memory, sample.sample_id)
        finally:
            memory_builder.close(memory)

    summary = {
        "run_id": config.run_id,
        "mode": "preembed",
        "sample_count": len(samples),
        "session_embedding_count": session_count,
        "turn_count": turn_count,
        "question_embedding_count": question_count,
        "cache": {
            "mode": "write",
            "cache_dir": str(config.embedding_cache_dir),
            "hits": cache_hits,
            "misses": cache_misses,
        },
        "config": config.to_jsonable(),
    }
    write_json(config.run_dir / "preembedding.json", summary)
    logger.info(
        "Preembedded embeddings samples={} sessions={} questions={} cache_hits={} cache_misses={}",
        len(samples),
        session_count,
        question_count,
        cache_hits,
        cache_misses,
    )
    return summary


def preembed_questions(memory: Any, sample: ConversationSample) -> int:
    embedder = memory_embedder(memory)
    for qa in sample.qa:
        embedder.embed(qa.question, "search")
    return len(sample.qa)
