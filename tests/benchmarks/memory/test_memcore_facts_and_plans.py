from __future__ import annotations

from typing import Any

import pytest

from benchmarks.common.vector_types import SearchHit
from benchmarks.memory.data import QuestionAnswer
from benchmarks.memory.facts import memory_fact_from_hit, text_hash
from benchmarks.memory.prompting import (
    EMPTY_MEMORY_TEXT,
    MEMORY_SYSTEM_PROMPT,
    MEMORY_TEMPLATE_PLACEHOLDER,
    build_query_messages,
    format_memory_fact,
)
from benchmarks.memory.types import RetrievedFact

MISSING = object()


def hit(
    memory_id: Any = "fact", text: Any = "A fact.", session: int = 1, **payload: Any
) -> SearchHit:
    fields: dict[str, Any] = {
        "memory": text,
        "created_at": "2023-01-01T00:00:00Z",
        "source_session_index": session,
        "source_session_id": f"session_{session}",
        "source_turn_index": 0,
        "source_turn_id": f"conv-1:session_{session}:0",
        **payload,
    }
    kept = {key: value for key, value in fields.items() if value is not MISSING}
    return SearchHit(id=memory_id, payload=kept, score=1.0, distance=0.0, rank=1)


def fact(memory_id: str, session: int = 1, text: str | None = None) -> RetrievedFact:
    return memory_fact_from_hit(hit(memory_id, text or f"Fact {memory_id}.", session))


def test_fact_is_read_from_the_promoted_payload_fields():
    retrieved = memory_fact_from_hit(
        hit(" fact-1 ", "  Alice likes tea.  ", 3, created_at=" 2023-01-03 ", source_turn_index="4")
    )

    assert retrieved == RetrievedFact(
        memory_id="fact-1",
        text="Alice likes tea.",
        text_hash=text_hash("Alice likes tea."),
        created_at="2023-01-03",
        source_session_index=3,
        source_session_id="session_3",
        source_turn_index=4,
        source_turn_id="conv-1:session_3:0",
    )
    assert text_hash("Alice likes tea.") == (
        "147d286e2e3b710fad83b40a6e7469980d7e228c2f873edc4fa464c15a8abc3b"
    )


@pytest.mark.parametrize("turn_id,turn_index", [("conv-1:session_1:12", 12)])
def test_turn_index_missing_from_the_payload_is_the_last_part_of_the_turn_id(turn_id, turn_index):
    retrieved = memory_fact_from_hit(hit(source_turn_index=MISSING, source_turn_id=turn_id))

    assert retrieved.source_turn_index == turn_index


def test_prompt_text_sent_to_the_model_is_pinned():
    question = QuestionAnswer("conv-1", "q0", "Where did Alice go?", "Lyon", "1")

    assert MEMORY_SYSTEM_PROMPT == (
        "You are a helpful assistant that remembers details from past conversations. "
        "Answer questions based on the conversation history provided. "
        "The following is a conversation history between two people:\n\n"
    )
    assert EMPTY_MEMORY_TEXT == "(No relevant memories found)\n"
    assert MEMORY_TEMPLATE_PLACEHOLDER == "<<<LOCOMO_JASPER_MEMORY_GOES_HERE>>>"
    assert build_query_messages(question) == [
        {
            "role": "user",
            "content": "Based on the conversation above, answer concisely.\n"
            "Question: Where did Alice go?",
        }
    ]
    assert format_memory_fact(fact("a", text="  Alice went to Lyon.  ")) == "Alice went to Lyon.\n"
