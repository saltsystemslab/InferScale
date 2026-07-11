from __future__ import annotations

import time
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
from typing import Any

import pytest

from locomo_jasper_bench.config import BenchmarkConfig
from locomo_jasper_bench.data import ConversationSample
from locomo_jasper_bench.embedding import preembed


def _sample(sample_id: str) -> ConversationSample:
    return ConversationSample(sample_id=sample_id, turns=[], qa=[], raw={})


def test_preembed_runs_samples_concurrently_and_aggregates_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    samples = [_sample("sample-1"), _sample("sample-2"), _sample("sample-3")]

    class FakeMemoryBuilder:
        barrier = Barrier(2)
        lock = Lock()
        started = 0
        active = 0
        max_active = 0
        closed: list[str] = []

        def __init__(self, config: BenchmarkConfig, *, embedding_cache_mode: str) -> None:
            del config, embedding_cache_mode

        def build_with_metrics(
            self,
            sample: ConversationSample,
            *,
            finalize_index: bool,
        ) -> tuple[Any, dict[str, int]]:
            assert finalize_index is False
            with self.lock:
                type(self).started += 1
                started = type(self).started
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            try:
                if started <= 2:
                    self.barrier.wait(timeout=5)
                time.sleep(0.02)
                return SimpleNamespace(sample_id=sample.sample_id), {
                    "memory_inferred_record_count": 1,
                }
            finally:
                with self.lock:
                    type(self).active -= 1

        def embedding_cache_stats(self, memory: Any) -> dict[str, int]:
            del memory
            return {"hits": 1, "misses": 2}

        def memory_llm_cache_stats(self, memory: Any) -> dict[str, int]:
            del memory
            return {"hits": 3, "misses": 4}

        def log_embedding_cache_stats(self, memory: Any, sample_id: str) -> None:
            del memory, sample_id

        def close(self, memory: Any) -> None:
            type(self).closed.append(str(memory.sample_id))

    monkeypatch.setattr(preembed, "load_locomo", lambda *args, **kwargs: samples)
    monkeypatch.setattr(preembed, "SampleMemoryBuilder", FakeMemoryBuilder)
    monkeypatch.setattr(preembed, "preembed_questions", lambda memory, sample: 2)
    monkeypatch.setattr(preembed, "preembed_question_entities", lambda memory, sample: 3)
    config = BenchmarkConfig(
        results_dir=tmp_path / "results",
        run_id="concurrent-preembed",
        preembed_workers=2,
    )

    summary = preembed.preembed_locomo_embeddings(config)

    assert FakeMemoryBuilder.max_active == 2
    assert sorted(FakeMemoryBuilder.closed) == ["sample-1", "sample-2", "sample-3"]
    assert summary["sample_count"] == 3
    assert summary["preembed_workers"] == 2
    assert summary["inferred_memory_count"] == 3
    assert summary["question_embedding_count"] == 6
    assert summary["entity_embedding_count"] == 9
    assert summary["cache"] == {
        "mode": "write",
        "cache_dir": str(config.embedding_cache_dir),
        "hits": 3,
        "misses": 6,
    }
    assert summary["memory_inference_cache"]["hits"] == 9
    assert summary["memory_inference_cache"]["misses"] == 12
    assert (config.run_dir / "preembedding.json").exists()


def test_preembed_rejects_duplicate_sample_ids_before_starting_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(preembed, "load_locomo", lambda *args, **kwargs: [_sample("same"), _sample("same")])
    config = BenchmarkConfig(results_dir=tmp_path, run_id="duplicate", preembed_workers=2)

    with pytest.raises(ValueError, match="Duplicate sample ids"):
        preembed.preembed_locomo_embeddings(config)

    assert not (config.run_dir / "preembedding.json").exists()
