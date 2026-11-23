from __future__ import annotations

from typing import Any

from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.mem0.memory_handles import added_memory_ids


def load_facts_into_memory(
    memory: Any, facts: tuple[MemoryFact, ...], *, link_entities: bool = True
) -> None:
    for fact in facts:
        result = memory.add(
            [{"role": "user", "content": fact.text}],
            user_id=fact.sample_id,
            infer=False,
            metadata=fact.metadata(),
        )
        if link_entities and added_memory_ids(result):
            memory._link_entities_for_memory(fact.id, fact.text, {"user_id": fact.sample_id})
