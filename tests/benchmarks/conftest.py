from __future__ import annotations

import pytest
from library_fakes import FakeEmbedder


@pytest.fixture
def local_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    embedder = FakeEmbedder()
    monkeypatch.setattr(
        "benchmarks.memory.mem0.local_embedder.shared_local_embedder",
        lambda model, device: embedder,
    )
    return embedder
