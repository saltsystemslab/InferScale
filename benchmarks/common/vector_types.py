"""Vector store selection shared by the Mem0 adapter and the retrieval baselines."""

from __future__ import annotations

from dataclasses import dataclass

from inferscale.v1.index.jasper import JasperIndexConfig
from inferscale.v1.types import VECTOR_DISTANCE, RetrievalMetrics, SearchHit, SearchMetrics

__all__ = [
    "VECTOR_DISTANCE",
    "RetrievalMetrics",
    "SearchHit",
    "SearchMetrics",
    "VectorStoreConfig",
]


@dataclass(slots=True)
class VectorStoreConfig:
    backend: str = "jasper"
    n_neighbors: int = 64
    alpha: float = 1.0
    workspace_budget: str = "10GB"
    beam_width: int = 64

    def jasper(self) -> JasperIndexConfig:
        return JasperIndexConfig(
            n_neighbors=self.n_neighbors,
            alpha=self.alpha,
            workspace_budget=self.workspace_budget,
            beam_width=self.beam_width,
        )
