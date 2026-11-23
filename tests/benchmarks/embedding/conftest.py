from __future__ import annotations

import functools
import itertools
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from benchmarks.embedding import model, run


@pytest.fixture(autouse=True)
def logging_configurations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    configurations: list[str] = []
    monkeypatch.setattr(run, "configure_logging", lambda: configurations.append("configured"))
    return configurations


@pytest.fixture
def embedding_log() -> Iterator[list[str]]:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="INFO")
    yield messages
    logger.remove(sink)


def _write_catalog(
    path: Path, sample_id: Any = "sample-1", texts: Any = ("A fact.",), **changes: Any
) -> Path:
    catalog = {
        "version": 5,
        "sample_id": sample_id,
        "model": "extractor",
        "embedding_model": "text-embedding-3-small",
        "facts": [
            {"id": str(index), "sample_id": sample_id, "text": text}
            for index, text in enumerate(texts)
        ],
        **changes,
    }
    path.write_text(json.dumps(catalog))
    return path


@pytest.fixture
def write_catalog() -> Callable[..., Path]:
    return _write_catalog


@pytest.fixture
def catalog_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "catalogs"
    folder.mkdir()
    _write_catalog(folder / "a.json", "sample-a", ["one fact", "another longer fact", "third"])
    _write_catalog(folder / "b.json", "sample-b", ["b fact one", "b"])
    return folder


class _RecordingEmbedder:
    def __init__(self, journal: list[tuple[str, Any]], **settings: Any) -> None:
        self.journal = journal
        self.metadata = {"model": settings["model"], "device": settings["device"]}
        journal.append(("model loaded", settings))

    def token_lengths(self, texts: list[str]) -> list[int]:
        return [len(text.split()) + 1 for text in texts]

    def encode(self, texts: list[str]) -> list[str]:
        self.journal.append(("encoded", list(texts)))
        return list(texts)

    def synchronize(self) -> None:
        pass

    def validate(self, vectors: list[str], count: int) -> None:
        assert len(vectors) == count

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory_bytes(self) -> int:
        return 4096


@pytest.fixture
def journal(
    monkeypatch: pytest.MonkeyPatch, embedding_log: list[str]
) -> Iterator[list[tuple[str, Any]]]:
    journal: list[tuple[str, Any]] = []
    monkeypatch.setattr(model, "QwenEmbedder", functools.partial(_RecordingEmbedder, journal))

    eighths_of_a_second = itertools.count(start=0.0, step=0.125)
    monkeypatch.setattr(
        run, "measure", functools.partial(run.measure, clock=lambda: next(eighths_of_a_second))
    )

    write_report = run.atomic_write_text

    def record_then_write_report(path: Path, text: str) -> None:
        journal.append(("report saved", json.loads(text)["complete"]))
        write_report(path, text)

    monkeypatch.setattr(run, "atomic_write_text", record_then_write_report)
    sink = logger.add(
        lambda message: journal.append(("logged", message.record["message"])), level="INFO"
    )
    yield journal
    logger.remove(sink)
