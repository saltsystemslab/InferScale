"""Fact extraction for one conversation: every turn is added to Mem0 with inference on."""

from __future__ import annotations

from typing import Any

from benchmarks.common.embedding_cache import CachedEmbedder
from benchmarks.memory.data import ConversationSample, Turn, format_turn_for_memory
from benchmarks.memory.mem0.fact_catalog import MemoryFact, make_memory_fact, source_metadata
from benchmarks.memory.mem0.mem0_patches import mem0_observation_date
from benchmarks.memory.mem0.memory_handles import added_memory_texts


def extract_facts(memory: Any, sample: ConversationSample, *, embedder: Any) -> list[MemoryFact]:
    """The deduplicated facts Mem0 infers from the conversation, each embedded once."""
    facts: list[MemoryFact] = []
    seen: set[str] = set()
    for turn in sample.turns:
        for text in _add_turn(memory, sample, turn):
            fact = make_memory_fact(text, sample, turn)
            if fact.id not in seen:
                facts.append(fact)
                seen.add(fact.id)
    _embed_final_texts(embedder, facts)
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


def _embed_final_texts(embedder: Any, facts: list[MemoryFact]) -> None:
    # Mem0's merge phase can store a fact text that differs from the string it
    # embedded (an update rewrites the text after the add-path embedding), so a
    # catalog replay would miss the read-only cache. Embedding every final text
    # makes catalogs replay-complete; cache hits make this free in the common case.
    for fact in facts:
        embedder.embed(fact.text, CachedEmbedder.DOCUMENT_PURPOSE)
