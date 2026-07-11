from __future__ import annotations

import math
import time
from typing import Any

from ..vector_types import RetrievalMetrics, SearchHit, SearchMetrics
from .fact_catalog import MemoryFact


class PreparedMem0Retriever:
    """Live Mem0 V3 retriever backed by an immutable inferred-fact catalog."""

    def __init__(
        self,
        memory: Any,
        *,
        sample_id: str,
        fact_catalog: tuple[MemoryFact, ...],
        vector_backend: str,
    ) -> None:
        self.memory = memory
        self.sample_id = sample_id
        self.fact_catalog = tuple(fact_catalog)
        self.vector_backend = vector_backend
        self._facts_by_id = {fact.id: fact for fact in self.fact_catalog}
        if len(self._facts_by_id) != len(self.fact_catalog):
            raise ValueError("Prepared Mem0 fact catalog contains duplicate ids.")

    def facts(self) -> tuple[MemoryFact, ...]:
        return self.fact_catalog

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> tuple[list[SearchHit], RetrievalMetrics]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        search = getattr(self.memory, "search", None)
        if not callable(search):
            raise RuntimeError("Prepared Mem0 memory has no callable search method.")

        original_embedder = getattr(self.memory, "embedding_model", None)
        timed_embedder = _TimedEmbedder(original_embedder) if original_embedder is not None else None
        if timed_embedder is not None:
            self.memory.embedding_model = timed_embedder
        started = time.perf_counter()
        try:
            result = search(
                query,
                top_k=top_k,
                filters={"user_id": self.sample_id},
            )
        finally:
            total_time_ms = (time.perf_counter() - started) * 1000
            if timed_embedder is not None:
                self.memory.embedding_model = original_embedder

        rows = result.get("results") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("Mem0 search did not return a results list.")
        if len(rows) > top_k:
            raise RuntimeError(
                f"Mem0 search returned {len(rows)} facts for top_k={top_k}; "
                "refusing to inject more facts than requested."
            )
        hits = [self._search_hit(row, rank) for rank, row in enumerate(rows, start=1)]
        ids = [hit.id for hit in hits]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Mem0 search returned duplicate stable fact ids.")

        store_metrics = _last_store_metrics(self.memory, self.vector_backend)
        return hits, RetrievalMetrics(
            embedding_time_ms=timed_embedder.elapsed_ms if timed_embedder is not None else 0.0,
            search_time_ms=store_metrics.search_time_ms,
            total_time_ms=total_time_ms,
            vector_backend=store_metrics.vector_backend,
            jasper_effective_beam_width=store_metrics.jasper_effective_beam_width,
        )

    def close(self) -> None:
        for store in (
            getattr(self.memory, "vector_store", None),
            getattr(self.memory, "_entity_store", None),
        ):
            close = getattr(store, "close", None)
            if callable(close):
                close()

    def _search_hit(self, row: Any, rank: int) -> SearchHit:
        if not isinstance(row, dict):
            raise RuntimeError(f"Mem0 search returned a non-object result at rank {rank}.")
        metadata = row.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        fact_id = str(metadata.get("fact_id") or "")
        if not fact_id or fact_id not in self._facts_by_id:
            raise RuntimeError(
                f"Mem0 search result at rank {rank} is not linked to the immutable fact catalog."
            )
        fact = self._facts_by_id[fact_id]
        text = str(row.get("memory") or fact.text)
        if text != fact.text:
            raise RuntimeError(
                f"Mem0 search text drifted from immutable fact {fact_id}: "
                f"catalog={fact.text!r} search={text!r}."
            )
        score = float(row.get("score") or 0.0)
        if not math.isfinite(score):
            raise RuntimeError(f"Mem0 search returned a non-finite score for fact {fact_id}.")
        promoted = fact.metadata()
        promoted.update(metadata)
        payload = {
            "memory": fact.text,
            "text": fact.text,
            "data": fact.text,
            "created_at": fact.created_at,
            **promoted,
            "metadata": promoted,
        }
        score_details = row.get("score_details")
        if isinstance(score_details, dict):
            payload["score_details"] = dict(score_details)
        return SearchHit(
            id=fact.id,
            payload=payload,
            score=score,
            distance=1.0 - score,
            rank=rank,
        )


class _TimedEmbedder:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self.elapsed_ms = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def embed(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._wrapped.embed(*args, **kwargs)
        finally:
            self.elapsed_ms += (time.perf_counter() - started) * 1000

    def embed_batch(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._wrapped.embed_batch(*args, **kwargs)
        finally:
            self.elapsed_ms += (time.perf_counter() - started) * 1000


def _last_store_metrics(memory: Any, backend: str) -> SearchMetrics:
    vector_store = getattr(memory, "vector_store", None)
    metrics = getattr(vector_store, "last_search_metrics", None)
    if isinstance(metrics, SearchMetrics):
        return metrics
    return SearchMetrics(
        search_time_ms=float(getattr(metrics, "search_time_ms", 0.0) or 0.0),
        vector_backend=getattr(metrics, "vector_backend", None) or backend,
        jasper_effective_beam_width=getattr(metrics, "jasper_effective_beam_width", None),
    )
