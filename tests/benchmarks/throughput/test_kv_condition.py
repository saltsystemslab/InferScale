from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.throughput import kv_condition, kv_requests
from benchmarks.memory.throughput.engine import StartedEngine
from benchmarks.memory.throughput.kv_corpus import CorpusStats
from benchmarks.memory.throughput.kv_requests import (
    ComposedRequests,
    RequestContext,
    RequestTimings,
    SampleRetrieval,
    compose_request,
    compose_requests,
)
from benchmarks.memory.throughput.projection import CorpusFootprint
from benchmarks.memory.throughput.stores import SampleStores, StoreSearch
from benchmarks.memory.throughput.workload import LocomoRequest
from inferscale.v1.kv.connector_utils import MEMORY_USER_ID_EXTRA_ARG
from inferscale.v1.types import KVChunk

SAMPLE = SimpleNamespace(sample_id="sample-0")
MEMORY_TOKENS = [1, 2, 5, 6, 7, 9]


def _chunk(chunk_id: str, token_ids: list[int]) -> KVChunk:
    return KVChunk(chunk_id=chunk_id, token_ids=token_ids, kv_by_layer={})


class _FakeComposer:
    def __init__(self) -> None:
        self.header_chunk = _chunk("__header__", [1, 2])
        self.footer_chunk = _chunk("__footer__", [9])
        self.chunk_store = object()
        self.chunks = {"f1": _chunk("f1", [5]), "f2": _chunk("f2", [6, 7])}
        self.encoder = SimpleNamespace(
            tokenizer=object(),
            compose=lambda chunks: {"layer": [chunk.chunk_id for chunk in chunks]},
        )


def _request(user_index: int) -> LocomoRequest:
    return LocomoRequest(user_index, "sample-0", f"q{user_index}", f"query{user_index}")


def _context(config, composer: _FakeComposer) -> RequestContext:
    corpus = SimpleNamespace(
        composer=lambda sample_id: composer,
        chunk_store=lambda sample_id: composer.chunk_store,
        tokenizer=composer.encoder.tokenizer,
    )
    retrieval = {"sample-0": SampleRetrieval(memory=object(), row_map=object())}
    return RequestContext(
        config=config, corpus=corpus, retrieval=retrieval, namespace="ns", scaffold=object()
    )


def _patch_compose_path(monkeypatch, *, canonical, registered, released) -> None:
    device_result = SimpleNamespace(stable_ids="ids")
    monkeypatch.setattr(kv_requests, "synchronize_cuda", lambda: None)
    monkeypatch.setattr(
        kv_requests,
        "search_store_for_kv",
        lambda memory, query, *, top_k: StoreSearch(device_result, 0.02, 0.003),
    )
    monkeypatch.setattr(
        kv_requests,
        "fetch_device_chunk",
        lambda store, stable_ids, row_map: _chunk("device-selection", [5, 6, 7]),
    )
    monkeypatch.setattr(
        kv_requests,
        "fetch_device_chunks",
        lambda store, meta, stable_ids, row_map, *, reverse: [meta["f1"], meta["f2"]],
    )
    monkeypatch.setattr(
        kv_requests, "release_chunks", lambda store, ids: released.append(list(ids))
    )
    monkeypatch.setattr(
        kv_requests,
        "register_user_memory",
        lambda namespace, **kwargs: registered.append((namespace, kwargs)),
    )
    monkeypatch.setattr(
        kv_requests, "materialize_store_device_result", lambda memory, result: ["hit"]
    )
    monkeypatch.setattr(
        kv_requests,
        "build_memory_prompt_token_ids",
        lambda tokenizer, sample, hits, **kwargs: SimpleNamespace(token_ids=canonical),
    )
    monkeypatch.setattr(
        kv_requests,
        "request_prompt_token_ids",
        lambda tokenizer, memory_token_ids, request: [*memory_token_ids, 100],
    )


@pytest.mark.parametrize("backend", ["gpu", "cpu"])
def test_compose_requests_charges_each_phase_and_routes_memories(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str
) -> None:
    config = make_throughput_config(
        tmp_path, user_counts=(2,), requests_per_user=1, kv_store_backend=backend
    )
    registered: list = []
    released: list = []
    _patch_compose_path(
        monkeypatch, canonical=MEMORY_TOKENS, registered=registered, released=released
    )
    context = _context(config, _FakeComposer())
    requests_by_user = {0: [_request(0)], 1: [_request(1)]}

    composed = compose_requests(context, requests_by_user, [SAMPLE], {"sample-0": ("f1", "f2")})

    assert composed.memory_user_ids == ["request-00000", "request-00001"]
    assert composed.prompts == [{"prompt_token_ids": [*MEMORY_TOKENS, 100]}] * 2
    assert composed.first_memory_user_id == "request-00000"
    assert composed.first_memory_token_ids == MEMORY_TOKENS
    assert composed.fact_count_total == 4
    assert composed.timings.retrieval_s == pytest.approx(0.04)
    assert composed.timings.vector_search_s == pytest.approx(0.006)
    assert [kwargs["user_id"] for _, kwargs in registered] == composed.memory_user_ids
    assert all(kwargs["token_ids"] == MEMORY_TOKENS for _, kwargs in registered)
    assert all(kwargs["num_tokens"] == len(MEMORY_TOKENS) for _, kwargs in registered)
    expected_release = [] if backend == "gpu" else ["f1", "f2"]
    assert released == [expected_release, expected_release]


def test_compose_request_rejects_memory_tokens_that_differ_from_prompt_injection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_throughput_config(tmp_path, kv_store_backend="gpu")
    _patch_compose_path(monkeypatch, canonical=[1, 2, 5, 9], registered=[], released=[])

    with pytest.raises(RuntimeError, match="differ"):
        compose_request(
            _context(config, _FakeComposer()),
            SAMPLE,
            _request(0),
            "request-00000",
            RequestTimings(),
        )


def test_compose_request_rejects_an_empty_cpu_selection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_throughput_config(tmp_path, kv_store_backend="cpu")
    _patch_compose_path(monkeypatch, canonical=MEMORY_TOKENS, registered=[], released=[])
    monkeypatch.setattr(kv_requests, "fetch_device_chunks", lambda *args, **kwargs: [])

    with pytest.raises(RuntimeError, match="no facts"):
        compose_request(
            _context(config, _FakeComposer()),
            SAMPLE,
            _request(0),
            "request-00000",
            RequestTimings(),
        )


def test_corpus_stats_sum_counters_and_weight_latencies_by_transfers() -> None:
    def store(host_mb, transfers, transferred, avg_ms, p95_ms, overlap, stall_ms):
        return SimpleNamespace(
            get_stats=lambda: {"total_host_mb": host_mb},
            get_bench_summary=lambda: {
                "total_transfers": transfers,
                "total_bytes_transferred": transferred,
                "avg_h2d_latency_ms": avg_ms,
                "p95_h2d_latency_ms": p95_ms,
                "avg_overlap_ratio": overlap,
                "total_staging_stall_ms": stall_ms,
            },
        )

    stats = CorpusStats.from_stores(
        [store(100.0, 1, 10, 2.0, 3.0, 0.5, 1.0), store(50.0, 3, 30, 4.0, 5.0, 0.9, 2.0)]
    )

    assert stats.host_mb == 150.0
    assert stats.h2d_bytes == 40
    assert stats.h2d_avg_ms == pytest.approx(3.5)
    assert stats.h2d_p95_ms == 5.0
    assert stats.h2d_overlap_ratio == pytest.approx(0.8)
    assert stats.staging_stall_ms == 3.0
    gpu_only = SimpleNamespace(get_stats=lambda: {}, get_bench_summary=lambda: {})
    assert CorpusStats.from_stores([gpu_only]) == CorpusStats(0.0, 0, 0.0, 0.0, 0.0, 0.0)


class _Params:
    def __init__(self, extra_args: dict | None = None) -> None:
        self.extra_args = extra_args

    def clone(self) -> _Params:
        return _Params(dict(self.extra_args) if self.extra_args is not None else None)


class _LLM:
    def __init__(self) -> None:
        self.calls: list[tuple[list, object]] = []

    def get_tokenizer(self):
        return SimpleNamespace(vocab_size=50)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls.append((prompts, sampling_params))
        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[SimpleNamespace(token_ids=[1, 2])],
            )
            for prompt in prompts
        ]

    def reset_prefix_cache(self) -> bool:
        return True


def test_measure_kv_batch_routes_requests_and_checks_coverage(tmp_path: Path) -> None:
    config = make_throughput_config(tmp_path, warmup_batches=1, kv_max_model_len=64)
    llm = _LLM()
    engine = StartedEngine(llm, _Params(), 0.1)
    composed = ComposedRequests(
        prompts=[{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5, 6, 7]}],
        memory_user_ids=["request-00000", "request-00001"],
        first_memory_user_id="request-00000",
        first_memory_token_ids=[1, 2],
    )
    resets = []
    coverage = {"requests_loaded": 2, "requests_covered": 2}
    connector = SimpleNamespace(
        reset_load_stats=lambda: resets.append(True), snapshot_load_stats=lambda: dict(coverage)
    )

    measured, load_stats = kv_condition.measure_kv_batch(
        config, engine, composed, llm.get_tokenizer(), connector
    )

    assert measured["total_input_tokens"] == 7
    assert measured["total_output_tokens"] == 4
    assert load_stats["requests_loaded"] == 2
    assert resets == [True]
    warmup_prompts, warmup_params = llm.calls[0]
    assert len(warmup_prompts) == 3
    assert warmup_prompts[2]["prompt_token_ids"][:2] == [1, 2]
    assert warmup_params[2].extra_args == {MEMORY_USER_ID_EXTRA_ARG: "request-00000"}
    measured_prompts, routed = llm.calls[1]
    assert measured_prompts == composed.prompts
    assert [
        params.extra_args[MEMORY_USER_ID_EXTRA_ARG] for params in routed
    ] == composed.memory_user_ids

    coverage["requests_covered"] = 1
    with pytest.raises(RuntimeError, match="covered 1 of 2"):
        kv_condition.measure_kv_batch(config, engine, composed, llm.get_tokenizer(), connector)


def test_run_kv_injection_requires_a_cuda_device(tmp_path: Path) -> None:
    config = make_throughput_config(tmp_path, kv_device="cpu", embedding_api_key="key")

    with pytest.raises(RuntimeError, match="cuda:0"):
        kv_condition.run_kv_injection(config, 1)


def test_run_kv_injection_frees_the_corpus_before_the_engine_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_throughput_config(
        tmp_path, run_id="kv-test", user_counts=(2,), requests_per_user=1, embedding_api_key="key"
    )
    events: list[str] = []
    corpus = SimpleNamespace(
        footprint=CorpusFootprint(
            fact_count=2, token_count=3, kv_bytes=300, scaffold_token_count=3
        ),
        probe_token_ids=[1],
        tokenizer=SimpleNamespace(vocab_size=100),
        precompute_time_s=4.0,
        stats=lambda: CorpusStats(1.0, 2, 3.0, 4.0, 0.5, 6.0),
        close=lambda: events.append("corpus_close"),
    )
    stores = SampleStores(memories={"sample-0": object()}, build_time_s={"sample-0": 2.0})
    composed = ComposedRequests(
        prompts=[{"prompt_token_ids": [1, 2, 3]}] * 2,
        memory_user_ids=["request-00000", "request-00001"],
        first_memory_user_id="request-00000",
        first_memory_token_ids=[1, 2],
        timings=RequestTimings(0.5, 0.1, 0.2, 0.3, 0.4, 0.6),
        fact_count_total=4,
    )
    llm = SimpleNamespace(get_tokenizer=lambda: object())

    def record(name: str, value=None):
        return lambda *args, **kwargs: events.append(name) or value

    monkeypatch.setattr(SampleStores, "close", lambda self: events.append("stores_close"))
    monkeypatch.setattr(kv_condition, "load_connector_module", lambda config: SimpleNamespace())
    monkeypatch.setattr(kv_condition, "force_vllm_inprocess_mode", lambda: None)
    monkeypatch.setattr(kv_condition, "load_samples", lambda config: [SAMPLE])
    monkeypatch.setattr(
        kv_condition, "build_locomo_requests", lambda *args, **kwargs: [_request(0), _request(1)]
    )
    monkeypatch.setattr(
        kv_condition,
        "fact_catalog_store",
        lambda config: SimpleNamespace(load=lambda sample: ("fact", "fact")),
    )
    monkeypatch.setattr(kv_condition, "get_gpu_memory_store", record("registry"))
    monkeypatch.setattr(kv_condition, "drop_namespace", record("drop_namespace"))
    monkeypatch.setattr(kv_condition, "namespace_stats", lambda namespace: {"total_gpu_mb": 64.0})
    monkeypatch.setattr(kv_condition, "precompute_corpus", record("precompute", corpus))
    monkeypatch.setattr(kv_condition, "check_kv_gpu_projection", record("projection"))
    monkeypatch.setattr(kv_condition, "prepare_sample_stores", record("stores", stores))
    monkeypatch.setattr(kv_condition, "build_sample_retrieval", lambda corpus, stores: ({}, 0.5))
    monkeypatch.setattr(
        kv_condition, "extract_memory_scaffold_token_ids", lambda tokenizer, *, block_size: object()
    )
    monkeypatch.setattr(kv_condition, "compose_requests", record("compose", composed))
    monkeypatch.setattr(
        kv_condition, "start_llm", record("start_llm", StartedEngine(llm, object(), 1.5))
    )
    monkeypatch.setattr(kv_condition, "require_tokenizer_parity", record("parity"))
    measured = {"generation_time_s": 2.0, "total_input_tokens": 20, "total_output_tokens": 10}
    monkeypatch.setattr(
        kv_condition,
        "measure_kv_batch",
        lambda *args: (measured, {"requests_loaded": 2, "requests_covered": 2}),
    )
    monkeypatch.setattr(kv_condition, "release_llm", record("release_llm"))

    row = kv_condition.run_kv_injection(config, 2)

    assert events.index("corpus_close") < events.index("start_llm")
    assert events[-4:] == ["stores_close", "corpus_close", "release_llm", "drop_namespace"]
    assert row["condition"] == "kv_injection"
    assert row["num_users"] == 2
    assert row["throughput_qps"] == 1.0
    assert row["fact_count"] == 2.0
    assert row["memory_setup_time_s"] == pytest.approx(2.5)
    assert row["kv_precompute_time_s"] == 4.0
    assert row["kv_compose_time_s"] == pytest.approx(0.2)
    assert row["kv_store_write_time_s"] == pytest.approx(0.3)
    assert row["kv_store_gpu_mb"] == 64.0
    assert row["kv_store_host_mb"] == 1.0
    assert row["kv_h2d_bytes"] == 2
    assert row["kv_requests_loaded"] == 2
    assert row["engine_startup_time_s"] == 1.5
