"""Retriever: embed and index chunk texts, embed and search queries."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from .protocols import Embedder, VectorIndex
from .types import RetrievalMetrics, SearchHit


class Retriever:
    def __init__(self, embedder: Embedder, index: VectorIndex, *, batch_size: int = 128) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self.embedder = embedder
        self.index = index
        self._batch_size = batch_size

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        payloads: Sequence[Mapping[str, Any]] | None = None,
    ) -> float:
        """Embed the texts in batches and add them to the index; returns the embedding time in ms."""
        id_list = [str(item_id) for item_id in ids]
        text_list = [str(text) for text in texts]
        if len(id_list) != len(text_list):
            raise ValueError("ids and texts must have the same length.")
        payload_list = (
            [dict(payload) for payload in payloads] if payloads is not None else [{} for _ in id_list]
        )
        if len(payload_list) != len(id_list):
            raise ValueError("payloads and ids must have the same length.")
        if not id_list:
            return 0.0
        started = time.perf_counter()
        vectors: list[list[float]] = []
        for start in range(0, len(text_list), self._batch_size):
            vectors.extend(self.embedder.embed_documents(text_list[start : start + self._batch_size]))
        embed_ms = (time.perf_counter() - started) * 1000
        self.index.add_many(vectors, payload_list, ids=id_list)
        return embed_ms

    def finalize(self) -> float:
        started = time.perf_counter()
        self.index.finalize()
        return (time.perf_counter() - started) * 1000

    def search(self, question: str, *, top_k: int) -> tuple[list[SearchHit], RetrievalMetrics]:
        embed_started = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        embedding_time_ms = (time.perf_counter() - embed_started) * 1000
        hits, search_metrics = self.index.search(query_vector, top_k)
        return hits, RetrievalMetrics(
            embedding_time_ms=embedding_time_ms,
            search_time_ms=search_metrics.search_time_ms,
            total_time_ms=embedding_time_ms + search_metrics.search_time_ms,
            vector_backend=search_metrics.vector_backend,
            jasper_effective_beam_width=search_metrics.jasper_effective_beam_width,
        )

    def close(self) -> None:
        self.index.close()
