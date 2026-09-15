"""Exact inner-product retrieval over a resident CUDA matrix."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..types import DeviceSearchResult, SearchHit, SearchMetrics
from .filters import payload_matches


class MatmulIndex:
    """Score every stored vector using FP32 CUDA matmul, then select top-k.

    Stable IDs are row ordinals; equal-score results may appear in any order.
    Torch and CUDA are required only when a nonempty corpus is finalized or searched.
    """

    def __init__(self, *, device: str = "cuda:0") -> None:
        if not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
            raise ValueError("MatmulIndex.device must be a CUDA device such as 'cuda:0'.")
        self.device = device
        self._vectors: np.ndarray | None = None
        self._vectors_gpu: Any = None
        self._rows: list[tuple[str, dict[str, Any]]] = []
        self._ordinals: dict[str, int] = {}
        self._finalized = False

    @property
    def vector_count(self) -> int:
        return len(self._rows)

    @property
    def dim(self) -> int | None:
        return None if self._vectors is None else int(self._vectors.shape[1])

    def add_many(
        self,
        vectors: Iterable[np.ndarray | list[float]],
        payloads: Iterable[dict[str, Any]],
        ids: Iterable[str] | None = None,
    ) -> list[str]:
        vector_list = [_as_vector(vector) for vector in vectors]
        payload_list = [dict(payload) for payload in payloads]
        id_list = [str(item_id) for item_id in ids] if ids is not None else [
            str(uuid.uuid4()) for _ in vector_list
        ]
        if len(vector_list) != len(payload_list):
            raise ValueError("vectors and payloads must have the same length")
        if len(id_list) != len(vector_list):
            raise ValueError("ids and vectors must have the same length")
        if len(set(id_list)) != len(id_list) or any(item_id in self._ordinals for item_id in id_list):
            raise ValueError("MatmulIndex vector IDs must be unique.")
        if not vector_list:
            return []
        dim = self.dim if self.dim is not None else int(vector_list[0].shape[0])
        if any(vector.shape[0] != dim for vector in vector_list):
            raise ValueError(f"Every vector dim must match store dim {dim}.")

        matrix = np.stack(vector_list)
        if self._vectors is not None:
            matrix = np.concatenate((self._vectors, matrix), axis=0)
        first_ordinal = self.vector_count
        self.close()
        self._vectors = matrix
        self._rows.extend(zip(id_list, payload_list))
        self._ordinals.update((item_id, first_ordinal + offset) for offset, item_id in enumerate(id_list))
        return id_list

    def finalize(self) -> None:
        if self._finalized:
            return
        if self._vectors is not None:
            torch = _require_cuda()
            self._vectors_gpu = torch.from_numpy(self._vectors).to(device=self.device, dtype=torch.float32)
        self._finalized = True

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[SearchHit], SearchMetrics]:
        result = self.search_device(query_vector, top_k, filters)
        if result is None:
            return [], SearchMetrics(0.0, vector_backend="exact")
        return self.materialize_device_result(result), result.metrics

    def search_device(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> DeviceSearchResult | None:
        if self._vectors is None:
            return None
        query = _as_vector(query_vector, name="query_vector")
        if query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} does not match store dim {self.dim}")
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
            raise ValueError("top_k must be an integer.")
        k = max(1, min(int(top_k), self.vector_count))
        eligible = None
        if filters:
            eligible = [ordinal for ordinal, (_, payload) in enumerate(self._rows) if payload_matches(payload, filters)]
            k = min(k, len(eligible))

        self.finalize()
        torch = _require_cuda()
        matrix = self._vectors_gpu
        if k == 0:
            return DeviceSearchResult(
                stable_ids=torch.empty(0, dtype=torch.int64, device=matrix.device),
                distances=torch.empty(0, dtype=torch.float32, device=matrix.device),
                metrics=SearchMetrics(0.0, vector_backend="exact"),
            )
        query_tensor = torch.from_numpy(query).to(device=matrix.device, dtype=torch.float32)
        eligible_tensor = None if eligible is None else torch.tensor(eligible, dtype=torch.int64, device=matrix.device)
        torch.cuda.synchronize(matrix.device)
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            scores = matrix @ query_tensor
            if eligible_tensor is not None:
                scores = scores.index_select(0, eligible_tensor)
            top_scores, top_ordinals = torch.topk(scores, k, sorted=True)
            if eligible_tensor is not None:
                top_ordinals = eligible_tensor.index_select(0, top_ordinals)
            distances = -top_scores
        torch.cuda.synchronize(matrix.device)
        search_time_ms = (time.perf_counter() - started) * 1000
        return DeviceSearchResult(
            stable_ids=top_ordinals.detach(),
            distances=distances.detach(),
            metrics=SearchMetrics(search_time_ms, vector_backend="exact"),
        )

    def materialize_device_result(self, result: DeviceSearchResult) -> list[SearchHit]:
        stable_ids = result.stable_ids.detach().cpu().tolist()
        distances = result.distances.detach().cpu().tolist()
        if len(stable_ids) != len(distances):
            raise RuntimeError("Exact result IDs and distances have different lengths.")
        hits: list[SearchHit] = []
        for stable_id, raw_distance in zip(stable_ids, distances):
            if not 0 <= stable_id < self.vector_count:
                raise RuntimeError(f"Exact search returned invalid stable vector id {stable_id}.")
            item_id, payload = self._rows[stable_id]
            distance = float(raw_distance)
            hits.append(SearchHit(item_id, dict(payload), -distance, distance, len(hits) + 1))
        return hits

    def stable_id_items(self) -> tuple[tuple[int, str], ...]:
        if not self._finalized:
            raise RuntimeError("Exact search stable IDs are unavailable before finalization.")
        return tuple((ordinal, item_id) for ordinal, (item_id, _) in enumerate(self._rows))

    def get(self, item_id: str) -> SearchHit | None:
        ordinal = self._ordinals.get(str(item_id))
        if ordinal is None:
            return None
        candidate_id, payload = self._rows[ordinal]
        return SearchHit(candidate_id, dict(payload), 1.0, 0.0, ordinal + 1)

    def close(self) -> None:
        self._vectors_gpu = None
        self._finalized = False


def _as_vector(vector: np.ndarray | list[float], *, name: str = "vector") -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _require_cuda() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Exact backend requires torch with CUDA support.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Exact backend requires an available CUDA device.")
    return torch
