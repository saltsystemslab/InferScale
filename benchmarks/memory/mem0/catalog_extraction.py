from __future__ import annotations

from typing import Any

from benchmarks.common.embedding_cache import CachedEmbedder
from benchmarks.memory.data import ConversationSample, Turn, format_turn_for_memory
from benchmarks.memory.mem0.fact_catalog import MemoryFact, make_memory_fact, source_metadata
from benchmarks.memory.mem0.mem0_patches import mem0_observation_date
from benchmarks.memory.mem0.memory_handles import added_memory_texts


def extract_facts(memory: Any, sample: ConversationSample, *, embedder: Any) -> list[MemoryFact]:
    facts: list[MemoryFact] = []
    seen_fact_ids: set[str] = set()
    for turn in sample.turns:
        for text in _add_turn(memory, sample, turn):
            fact = make_memory_fact(text, sample, turn)
            if fact.id not in seen_fact_ids:
                facts.append(fact)
                seen_fact_ids.add(fact.id)
    _embed_final_texts_for_replay(embedder, facts)
    return facts


def _add_turn(memory: Any, sample: ConversationSample, turn: Turn) -> list[str]:
    metadata = source_metadata(sample, turn)
    with mem0_observation_date(str(metadata["created_at"])):
        result = memory.add(
            [{"role": str(metadata["role"]), "content": format_turn_for_memory(turn)}],
            user_id=sample.sample_id,
            infer=True,
            metadata=metadata,
        )
    return added_memory_texts(result)


def _embed_final_texts_for_replay(embedder: Any, facts: list[MemoryFact]) -> None:
    for fact in facts:
        embedder.embed(fact.text, CachedEmbedder.DOCUMENT_PURPOSE)
