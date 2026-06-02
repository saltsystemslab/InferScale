from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .clients import EmbeddingClient
from .jasper_store import JasperVectorStore, SearchHit, SearchMetrics


@dataclass(slots=True)
class MemorySearch:
    hits: list[SearchHit]
    embedding_time_ms: float
    store_metrics: SearchMetrics

    @property
    def total_time_ms(self) -> float:
        return self.embedding_time_ms + self.store_metrics.search_time_ms


class JasperMemory:
    """Small Mem0-like memory wrapper around embeddings and JasperVectorStore."""

    def __init__(self, *, embedder: EmbeddingClient, store: JasperVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def add(
        self,
        messages: str | list[dict[str, Any]],
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        texts = _messages_to_texts(messages)
        payloads = [
            {
                "memory": text,
                "user_id": user_id,
                "metadata": metadata or {},
            }
            for text in texts
        ]
        return self.add_texts(texts, payloads)

    def add_texts(
        self,
        texts: list[str],
        payloads: list[dict[str, Any]],
        ids: list[str] | None = None,
    ) -> list[str]:
        vectors = self.embedder.embed(texts)
        return self.store.add_many(vectors, payloads, ids)

    def search(self, query: str, *, top_k: int = 10, user_id: str | None = None) -> list[dict[str, Any]]:
        result = self.search_with_metrics(query, top_k=top_k, user_id=user_id)
        return [_hit_to_mem0_result(hit) for hit in result.hits]

    def search_with_metrics(self, query: str, *, top_k: int = 10, user_id: str | None = None) -> MemorySearch:
        started = time.perf_counter()
        query_vector = self.embedder.embed([query])[0]
        embedding_time_ms = (time.perf_counter() - started) * 1000
        hits, metrics = self.store.search(query_vector, top_k=top_k)
        if user_id is not None:
            hits = [hit for hit in hits if hit.payload.get("user_id") == user_id or hit.payload.get("metadata", {}).get("user_id") == user_id]
        return MemorySearch(hits=hits, embedding_time_ms=embedding_time_ms, store_metrics=metrics)


def _messages_to_texts(messages: str | list[dict[str, Any]]) -> list[str]:
    if isinstance(messages, str):
        return [messages]
    texts: list[str] = []
    for message in messages:
        role = str(message.get("role") or message.get("speaker") or "").strip()
        content = str(message.get("content") or message.get("text") or "").strip()
        if not content:
            continue
        texts.append(f"{role}: {content}" if role else content)
    return texts


def _hit_to_mem0_result(hit: SearchHit) -> dict[str, Any]:
    payload = hit.payload
    return {
        "id": hit.id,
        "memory": payload.get("memory") or payload.get("text") or "",
        "score": hit.score,
        "metadata": payload.get("metadata", {}),
    }
