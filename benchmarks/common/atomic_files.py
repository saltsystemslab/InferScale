from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import numpy as np

COMPACT_SEPARATORS = (",", ":")


@contextmanager
def _replacing(destination: Path, mode: str) -> Iterator[IO[Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    is_text_mode = "b" not in mode
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode=mode,
            encoding="utf-8" if is_text_mode else None,
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as replacement:
            temporary_path = Path(replacement.name)
            yield replacement
            replacement.flush()
            os.fsync(replacement.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    with _replacing(path, "wb") as replacement:
        replacement.write(data)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any, *, indent: int | None = None) -> None:
    with _replacing(path, "w") as replacement:
        json.dump(
            payload,
            replacement,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
            separators=COMPACT_SEPARATORS if indent is None else None,
        )
        replacement.write("\n")


def atomic_write_npy(path: Path, array: np.ndarray) -> None:
    with _replacing(path, "wb") as replacement:
        np.save(replacement, array.astype(np.float32, copy=False))
