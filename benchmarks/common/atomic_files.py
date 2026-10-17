"""Atomic file writes: a temporary file beside the target, fsync, then rename.

Readers of caches and result files never see a partially written file, and a
crash leaves the previous contents in place.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import numpy as np


@contextmanager
def _replacing(path: Path, mode: str) -> Iterator[IO[Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            encoding="utf-8" if "b" not in mode else None,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    with _replacing(path, "wb") as handle:
        handle.write(data)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any, *, indent: int | None = None) -> None:
    """Write JSON with sorted keys, compact when ``indent`` is None, and a trailing newline."""
    with _replacing(path, "w") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )
        handle.write("\n")


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    with _replacing(path, "wb") as handle:
        np.save(handle, array.astype(np.float32, copy=False))
