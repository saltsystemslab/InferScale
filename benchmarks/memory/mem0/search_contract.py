"""Checks every vector backend's search results must pass before the benchmark uses them."""

from __future__ import annotations

import numpy as np

from benchmarks.common.vector_types import VECTOR_DISTANCE, SearchHit

# Jasper fills result slots it could not populate with distances of this
# magnitude; such a hit is a sign of incomplete retrieval, not a neighbor.
SENTINEL_MAGNITUDE = 1e30


def validate_search_hits(hits: list[SearchHit], *, expected_count: int, backend: str) -> None:
    if len(hits) != expected_count:
        raise RuntimeError(
            f"{backend} returned {len(hits)} hits, expected {expected_count}; "
            "refusing to use incomplete retrieval."
        )
    ids = [hit.id for hit in hits]
    if len(ids) != len(set(ids)):
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
        if abs(hit.distance) >= SENTINEL_MAGNITUDE or abs(hit.score) >= SENTINEL_MAGNITUDE:
            raise RuntimeError(
                f"{backend} returned a sentinel-like score or distance for result {hit.id!r}."
            )


def require_inner_product(distance: str) -> None:
    if distance != VECTOR_DISTANCE:
        raise ValueError(
            f"Mem0 Jasper provider requires distance={VECTOR_DISTANCE!r}, got {distance!r}."
        )


def first_vector(vectors: np.ndarray | list[float] | list[list[float]]) -> np.ndarray:
    """The query vector Mem0 passed, which it wraps in a list for single queries."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[0] > 0:
        return array[0]
    raise ValueError("vectors must be a one-dimensional vector or a non-empty list of vectors")
