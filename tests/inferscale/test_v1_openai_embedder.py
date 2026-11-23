from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from inferscale.v1.embedding.openai import OpenAIEmbedder


class EmbeddingService:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def client(self, **options: Any) -> SimpleNamespace:
        return SimpleNamespace(embeddings=SimpleNamespace(create=self.create))

    def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        rows = [
            SimpleNamespace(index=position, embedding=(float(len(text)), float(position)))
            for position, text in enumerate(request["input"])
        ]
        returned_in_reverse = rows[::-1]
        return SimpleNamespace(data=returned_in_reverse)


@pytest.fixture
def service(monkeypatch) -> EmbeddingService:
    service = EmbeddingService()
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=service.client))
    return service


def test_vectors_follow_the_order_of_the_texts_whatever_order_the_service_answers_in(
    service,
) -> None:
    embedder = OpenAIEmbedder(model="embedding-model")

    vectors = embedder.embed_documents(["a", "bbb", "cc"])

    assert vectors == [[1.0, 0.0], [3.0, 1.0], [2.0, 2.0]]
    assert service.requests == [{"model": "embedding-model", "input": ["a", "bbb", "cc"]}]
