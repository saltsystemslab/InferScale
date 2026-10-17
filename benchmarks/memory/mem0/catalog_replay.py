"""Replay an immutable fact catalog into a fresh Mem0 store without inference."""

from __future__ import annotations

from typing import Any

from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.mem0.memory_handles import added_memory_ids


def load_facts_into_memory(
    memory: Any, facts: tuple[MemoryFact, ...], *, link_entities: bool = True
) -> None:
    """Add every catalog fact with ``infer=False`` so Mem0 stores the text verbatim.

    ``link_entities=False`` skips the entity store, which only Mem0's own
    search reads; callers that query the vector store directly never need it.
    """
    for fact in facts:
        result = memory.add(
            [{"role": "user", "content": fact.text}],
            user_id=fact.sample_id,
            infer=False,
            metadata=fact.metadata(),
        )
        if link_entities and added_memory_ids(result):
            # Mem0 links entities inside ``add`` only on the inference path.
            memory._link_entities_for_memory(fact.id, fact.text, {"user_id": fact.sample_id})
