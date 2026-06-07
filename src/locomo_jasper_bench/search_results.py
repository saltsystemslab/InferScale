from __future__ import annotations

from .mem0_adapter import Mem0JasperSearchResult
from .vector_types import SearchHit


def mem0_results_to_search_hits(results: list[Mem0JasperSearchResult]) -> list[SearchHit]:
    return [
        SearchHit(
            id=str(item.id),
            payload=dict(item.payload),
            score=float(item.score),
            distance=float(item.score),
            rank=rank,
        )
        for rank, item in enumerate(results, start=1)
    ]
