from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.memory.throughput import mem0_condition
from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.workload import LocomoRequest


def _run_with_search_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metrics: list[object],
    *,
    backend: str = "qdrant",
) -> list[dict]:
    config = ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
        run_id="diagnostics-test",
        user_counts=(1, 2, 3),
        requests_per_user=2,
        embedding_api_key="test-key",
        embedding_cache_dir=tmp_path / "embeddings",
        memory_llm_cache_dir=tmp_path / "llm",
        local_store_dir=tmp_path / "stores",
    )
    samples = [SimpleNamespace(sample_id=f"sample-{index}") for index in range(2)]
    requests = [
        LocomoRequest(
            user_id=f"user-{index}",
            user_index=index,
            sample_id=samples[index % 2].sample_id,
            question_id=f"question-{index}-{request_index}",
            query=f"query-{index}-{request_index}",
        )
        for index in range(3)
        for request_index in range(2)
    ]
    stores: dict[str, SimpleNamespace] = {}
    closed = []
    searched = []
    measured_sizes = []
    llm = SimpleNamespace(get_tokenizer=lambda: object())

    def build_store(config, *, backend, store_root, facts):
        store = SimpleNamespace(vector_store=SimpleNamespace())
        stores[store_root.name] = store
        return store

    def search(memory, query, *, top_k):
        assert top_k == config.top_k
        memory.vector_store.last_search_metrics = metrics[len(searched)]
        searched.append((memory, query))
        return [], 0.02, 0.003

    def measure(llm, prompts, sampling_params):
        measured_sizes.append(len(prompts))
        return {
            "generation_time_s": 2.0,
            "total_input_tokens": 10 * len(prompts),
            "total_output_tokens": 2 * len(prompts),
        }

    monkeypatch.setattr(mem0_condition, "load_samples", lambda config: samples)
    monkeypatch.setattr(
        mem0_condition, "fact_catalog_store", lambda config: SimpleNamespace(load=lambda sample: ())
    )
    monkeypatch.setattr(mem0_condition, "build_locomo_requests", lambda *args, **kwargs: requests)
    monkeypatch.setattr(mem0_condition, "build_user_store", build_store)
    monkeypatch.setattr(mem0_condition, "search_store", search)
    monkeypatch.setattr(mem0_condition, "start_llm", lambda config: (llm, object(), 0.1))
    monkeypatch.setattr(mem0_condition, "release_llm", lambda llm: None)
    monkeypatch.setattr(mem0_condition, "close_mem0", closed.append)
    monkeypatch.setattr(mem0_condition, "extract_memory_scaffold_token_ids", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        mem0_condition,
        "build_memory_prompt_token_ids",
        lambda *args, **kwargs: SimpleNamespace(token_ids=[1, 2]),
    )
    monkeypatch.setattr(
        mem0_condition,
        "build_kv_equivalence_prompt_token_ids",
        lambda *args, **kwargs: SimpleNamespace(prompt_token_ids=[1, 2, 3]),
    )
    monkeypatch.setattr(mem0_condition, "validate_prompt_lengths", lambda *args: None)
    monkeypatch.setattr(mem0_condition, "warm_up", lambda *args, **kwargs: None)
    monkeypatch.setattr(mem0_condition, "measure_batch", measure)

    rows = mem0_condition.run_mem0(
        config, config.user_counts, condition=f"mem0_{backend}", backend=backend
    )

    assert [query for memory, query in searched] == [request.query for request in requests]
    assert all(memory is stores[request.sample_id] for (memory, _), request in zip(searched, requests))
    assert len(closed) == len(stores) == 2
    assert measured_sizes == [2, 4, 6]
    for count, row in enumerate(rows, start=1):
        assert row["retrieval_time_s"] == pytest.approx(count * 0.04)
        assert row["vector_search_time_s"] == pytest.approx(count * 0.006)
        assert row["generation_time_s"] == 2.0
        assert row["throughput_qps"] == float(count)
    return rows


def test_deepcopy_diagnostics_accumulate_for_each_applicable_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def diagnostics(time_ms, calls):
        return SimpleNamespace(qdrant_deepcopy_time_ms=time_ms, qdrant_deepcopy_calls=calls)

    rows = _run_with_search_metrics(
        monkeypatch,
        tmp_path,
        [
            None,
            diagnostics(0.0, 0),
            diagnostics(1.25, 2),
            diagnostics(None, None),
            diagnostics(4.5, 3),
            SimpleNamespace(search_time_ms=3.0),
        ],
    )

    assert [row["qdrant_deepcopy_time_ms"] for row in rows] == [0.0, 1.25, 5.75]
    assert [row["qdrant_deepcopy_calls"] for row in rows] == [0, 2, 5]


@pytest.mark.parametrize("backend", ["qdrant", "jasper"])
def test_unavailable_deepcopy_diagnostics_remain_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str
) -> None:
    rows = _run_with_search_metrics(
        monkeypatch,
        tmp_path,
        [
            None,
            SimpleNamespace(search_time_ms=3.0),
            SimpleNamespace(qdrant_deepcopy_time_ms=None, qdrant_deepcopy_calls=None),
        ] * 2,
        backend=backend,
    )

    assert all(row["qdrant_deepcopy_time_ms"] is None for row in rows)
    assert all(row["qdrant_deepcopy_calls"] is None for row in rows)
