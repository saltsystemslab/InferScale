from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from benchmarks.common.atomic_files import COMPACT_SEPARATORS, atomic_write_text


class JsonlWriter(AbstractContextManager["JsonlWriter"]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    serialized = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    atomic_write_text(Path(path), serialized + "\n")


def read_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    saved = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        return default
    return saved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def replace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: csv_value(row.get(column)) for column in columns})
    atomic_write_text(Path(path), buffer.getvalue())


def csv_value(value: Any) -> Any:
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False, separators=COMPACT_SEPARATORS)
    if isinstance(value, bool):
        return int(value)
    return value
