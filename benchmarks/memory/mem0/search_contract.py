from __future__ import annotations

import numpy as np

from benchmarks.common.vector_types import VECTOR_DISTANCE, SearchHit
from inferscale.v1.index.jasper import PADDING_DISTANCE_MAGNITUDE


def validate_search_hits(hits: list[SearchHit], *, expected_count: int, backend: str) -> None:
    if len(hits) != expected_count:
        raise RuntimeError(
            f"{backend} returned {len(hits)} hits, expected {expected_count}; "
            "refusing to use incomplete retrieval."
        )
    hit_ids = [hit.id for hit in hits]
    if len(hit_ids) != len(set(hit_ids)):
        raise RuntimeError(
            f"{backend} returned duplicate result ids; refusing to use corrupt retrieval."
        )
    ranks = [hit.rank for hit in hits]
    if ranks != list(range(1, expected_count + 1)):
        raise RuntimeError(
            f"{backend} returned non-contiguous ranks {ranks[:10]!r}; expected ranks starting at 1."
        )
    for hit in hits:
        if not np.isfinite(hit.distance) or not np.isfinite(hit.score):
            raise RuntimeError(
                f"{backend} returned a non-finite score or distance for result {hit.id!r}."
            )
        if (
            abs(hit.distance) >= PADDING_DISTANCE_MAGNITUDE
            or abs(hit.score) >= PADDING_DISTANCE_MAGNITUDE
        ):
            raise RuntimeError(
                f"{backend} returned a sentinel-like score or distance for result {hit.id!r}."
            )


def require_inner_product(distance: str) -> None:
    if distance != VECTOR_DISTANCE:
        raise ValueError(
            f"Mem0 Jasper provider requires distance={VECTOR_DISTANCE!r}, got {distance!r}."
        )


def first_vector(vectors: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    vector_or_batch = np.asarray(vectors, dtype=np.float32)
    if vector_or_batch.ndim == 1:
        return vector_or_batch
    if vector_or_batch.ndim == 2 and vector_or_batch.shape[0] > 0:
        return vector_or_batch[0]
    raise ValueError("vectors must be a one-dimensional vector or a non-empty list of vectors")
