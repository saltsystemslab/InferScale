from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from locomo_jasper_bench.data import ConversationSample, QuestionAnswer
from locomo_jasper_bench.embedding import preembed


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.batch_calls: list[tuple[list[str], str]] = []

    def embed(self, text: Any, purpose: str | None = None) -> list[float]:
        del purpose
        return [float(len(str(text)))]

    def embed_batch(self, texts: list[str], purpose: str | None = None) -> list[list[float]]:
        self.batch_calls.append((list(texts), str(purpose)))
        return [[float(len(text))] for text in texts]


def _memory(embedder: _RecordingEmbedder) -> Any:
    return SimpleNamespace(
        embedding_model=embedder,
        _normalize_entity_text=lambda text: text.strip().casefold(),
    )


def _sample(questions: list[str]) -> ConversationSample:
    return ConversationSample(
        sample_id="sample-1",
        turns=[],
        qa=[
            QuestionAnswer(
                sample_id="sample-1",
                question_id=f"q{index}",
                question=question,
                answer="",
                category="1",
            )
            for index, question in enumerate(questions)
        ],
        raw={},
    )


def test_question_entities_are_deduped_like_mem0_and_batch_embedded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities_by_question = {
        "Where did Caroline go?": [
            ("PERSON", "Caroline"),
            ("PERSON", "caroline"),  # dedupes via normalization
            ("GPE", "Paris"),
        ],
        "What happened?": [],
    }
    monkeypatch.setattr(
        preembed,
        "_extract_query_entities",
        lambda question: entities_by_question[question],
    )
    embedder = _RecordingEmbedder()

    count = preembed.preembed_question_entities(
        _memory(embedder),
        _sample(list(entities_by_question)),
    )

    assert count == 2
    assert embedder.batch_calls == [(["Caroline", "Paris"], "search")]


def test_entity_dedup_caps_at_first_eight_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    many = [("THING", f"entity-{index}") for index in range(12)]
    monkeypatch.setattr(preembed, "_extract_query_entities", lambda question: many)
    embedder = _RecordingEmbedder()

    count = preembed.preembed_question_entities(_memory(embedder), _sample(["q"]))

    assert count == 8
    assert embedder.batch_calls[0][0] == [f"entity-{index}" for index in range(8)]


def test_memory_without_mem0_normalizer_is_rejected() -> None:
    memory = SimpleNamespace(embedding_model=_RecordingEmbedder())

    with pytest.raises(RuntimeError, match="_normalize_entity_text"):
        preembed.preembed_question_entities(memory, _sample(["q"]))
