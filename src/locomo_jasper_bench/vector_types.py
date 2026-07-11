from __future__ import annotations

from dataclasses import dataclass
from typing import Any


VECTOR_DISTANCE = "ip"


@dataclass(slots=True)
class VectorStoreConfig:
    backend: str = "jasper"
    n_neighbors: int = 64
    alpha: float = 1.0
    workspace_budget: str = "10GB"
    beam_width: int = 64


@dataclass(slots=True)
class SearchMetrics:
    search_time_ms: float
    vector_backend: str | None = None
    jasper_effective_beam_width: int | None = None


@dataclass(slots=True)
class RetrievalMetrics:
    embedding_time_ms: float
    search_time_ms: float
    total_time_ms: float
    vector_backend: str | None = None
    jasper_effective_beam_width: int | None = None


@dataclass(slots=True)
class SearchHit:
    id: str
    payload: dict[str, Any]
    score: float
    distance: float
    rank: int
