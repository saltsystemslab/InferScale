from __future__ import annotations

from typing import Any

import numpy as np


def as_float32_vector(vector: Any, *, name: str = "vector") -> np.ndarray:
    array = np.ascontiguousarray(vector, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)
