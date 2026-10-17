"""Retrieval through Mem0's own search over a store built from an immutable fact catalog."""

from __future__ import annotations

import math
import time
from typing import Any

from benchmarks.common.vector_types import RetrievalMetrics, SearchHit
from benchmarks.memory.mem0.fact_catalog import fact_payload
from benchmarks.memory.mem0.memory_handles import BuiltMemory


class PreparedMem0Retriever:
    def __init__(self, built: BuiltMemory, *, sample_id: str, vector_backend: str) -> None:
        self.built = built
        self.sample_id = sample_id
        self.vector_backend = vector_backend
        self._facts_by_id = {fact.id: fact for fact in built.facts}
        if len(self._facts_by_id) != len(built.facts):
            raise ValueError("Prepared Mem0 fact catalog contains duplicate ids.")

    @property
    def memory(self) -> Any:
        return self.built.memory

    def search(self, query: str, *, top_k: int) -> tuple[list[SearchHit], RetrievalMetrics]:
        """Mem0's search results as catalog-backed hits, best first."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")
        embed_ms_before = self.built.embed_time_ms
        started = time.perf_counter()
        result = self.memory.search(query, top_k=top_k, filters={"user_id": self.sample_id})
        total_time_ms = (time.perf_counter() - started) * 1000
        hits = [self._search_hit(row, rank) for rank, row in enumerate(_rows(result, top_k), 1)]
        if len({hit.id for hit in hits}) != len(hits):
            raise RuntimeError("Mem0 search returned duplicate stable fact ids.")
        store_metrics = self.memory.vector_store.last_search_metrics
        return hits, RetrievalMetrics(
            embedding_time_ms=self.built.embed_time_ms - embed_ms_before,
            search_time_ms=store_metrics.search_time_ms,
            total_time_ms=total_time_ms,
            vector_backend=store_metrics.vector_backend or self.vector_backend,
            jasper_effective_beam_width=store_metrics.jasper_effective_beam_width,
        )

    def log_cache_stats(self) -> None:
        self.built.log_cache_stats(self.sample_id)

    def close(self) -> None:
        self.built.close()

    def _search_hit(self, row: Any, rank: int) -> SearchHit:
        if not isinstance(row, dict):
            raise RuntimeError(f"Mem0 search returned a non-object result at rank {rank}.")
        metadata = row.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        fact = self._facts_by_id.get(str(metadata.get("fact_id") or ""))
        if fact is None:
            raise RuntimeError(
                f"Mem0 search result at rank {rank} is not linked to the immutable fact catalog."
            )
        text = str(row.get("memory") or fact.text)
        if text != fact.text:
            raise RuntimeError(
                f"Mem0 search text drifted from immutable fact {fact.id}: "
                f"catalog={fact.text!r} search={text!r}."
            )
        score = float(row.get("score") or 0.0)
        if not math.isfinite(score):
            raise RuntimeError(f"Mem0 search returned a non-finite score for fact {fact.id}.")
        payload = fact_payload(fact, {**fact.metadata(), **metadata})
        score_details = row.get("score_details")
        if isinstance(score_details, dict):
            payload["score_details"] = dict(score_details)
        return SearchHit(id=fact.id, payload=payload, score=score, distance=1.0 - score, rank=rank)


def _rows(result: Any, top_k: int) -> list[Any]:
    rows = result.get("results") if isinstance(result, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("Mem0 search did not return a results list.")
    if len(rows) > top_k:
        raise RuntimeError(
            f"Mem0 search returned {len(rows)} facts for top_k={top_k}; "
            "refusing to inject more facts than requested."
        )
    return rows
