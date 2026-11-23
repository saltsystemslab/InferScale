from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.throughput import kv_condition
from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.engine import StartedEngine
from benchmarks.memory.throughput.kv_corpus import CorpusStats
from benchmarks.memory.throughput.kv_requests import (
    ComposedRequests,
    RequestTimings,
    SampleRetrieval,
)
from benchmarks.memory.throughput.projection import CorpusFootprint
from benchmarks.memory.throughput.stores import SampleStores
from benchmarks.memory.throughput.workload import LocomoRequest
from inferscale.v1.kv.connector_utils import MEMORY_USER_ID_EXTRA_ARG
from inferscale.v1.kv.load_stats import LoadStats

SAMPLES = [SimpleNamespace(sample_id="sample-0"), SimpleNamespace(sample_id="sample-1")]
FOOTPRINT = CorpusFootprint(fact_count=2, token_count=3, kv_bytes=300, scaffold_token_count=3)
CORPUS_STATS = CorpusStats(1.0, 2, 3.0, 4.0, 0.5, 6.0)
MEASURED = {"generation_time_s": 2.0, "total_input_tokens": 20, "total_output_tokens": 10}
LOAD_STATS = {"requests_loaded": 2, "requests_covered": 2}


class KVRun:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **settings: Any) -> None:
        self.calls: dict[str, tuple[tuple, dict[str, Any]]] = {}
        self.config = make_throughput_config(
            tmp_path,
            **{
                "run_id": "kv-phases",
                "user_counts": (2,),
                "requests_per_user": 1,
                "embedding_api_key": "key",
                **settings,
            },
        )
        self.retrieval = {
            "sample-0": SampleRetrieval(memory=object(), row_map=object()),
            "sample-1": SampleRetrieval(memory=object(), row_map=object()),
        }
        self.row_maps_when_the_corpus_first_closed: int | None = None
        self.corpus = SimpleNamespace(
            footprint=FOOTPRINT,
            probe_token_ids=[1],
            tokenizer=SimpleNamespace(vocab_size=100),
            precompute_time_s=4.0,
            stats=self.phase("corpus.stats", CORPUS_STATS),
            close=self.close_corpus,
        )
        self.stores = SampleStores(
            memories={"sample-0": object(), "sample-1": object()},
            build_time_s={"sample-0": 2.0, "sample-1": 0.75},
        )
        self.composed = ComposedRequests(
            prompts=[{"prompt_token_ids": [1, 2, 3]}] * 2,
            memory_user_ids=["request-00000", "request-00001"],
            first_memory_user_id="request-00000",
            first_memory_token_ids=[1, 2],
            timings=RequestTimings(0.5, 0.125, 0.25, 0.375, 0.4375, 0.625),
            fact_count_total=6,
        )
        self.engine = StartedEngine(SimpleNamespace(get_tokenizer=object), object(), 1.5)
        monkeypatch.setattr(SampleStores, "close", lambda stores: None)
        monkeypatch.setattr(kv_condition.uuid, "uuid4", lambda: SimpleNamespace(hex="cafe"))
        replacements = {
            "load_connector_module": self.phase("load_connector_module", SimpleNamespace()),
            "force_vllm_inprocess_mode": self.phase("force_vllm_inprocess_mode"),
            "load_samples": self.phase("load_samples", SAMPLES),
            "build_locomo_requests": self.build_requests,
            "fact_catalog_store": lambda config: SimpleNamespace(
                load=lambda sample: ("fact", "fact", "fact")
            ),
            "get_gpu_memory_store": self.phase("get_gpu_memory_store"),
            "drop_namespace": self.phase("drop_namespace"),
            "namespace_stats": self.phase("namespace_stats", {"total_gpu_mb": 64}),
            "precompute_corpus": self.phase("precompute_corpus", self.corpus),
            "check_kv_gpu_projection": self.phase("check_kv_gpu_projection"),
            "prepare_sample_stores": self.phase("prepare_sample_stores", self.stores),
            "build_sample_retrieval": self.phase("build_sample_retrieval", (self.retrieval, 0.5)),
            "extract_memory_scaffold_token_ids": lambda tokenizer, *, block_size: "scaffold",
            "compose_requests": self.phase("compose_requests", self.composed),
            "start_llm": self.phase("start_llm", self.engine),
            "require_tokenizer_parity": self.phase("require_tokenizer_parity"),
            "measure_kv_batch": self.phase("measure_kv_batch", (MEASURED, LOAD_STATS)),
            "release_llm": self.phase("release_llm"),
        }
        for name, replacement in replacements.items():
            monkeypatch.setattr(kv_condition, name, replacement)

    def phase(self, name: str, result: Any = None):
        def run_phase(*args: Any, **kwargs: Any) -> Any:
            self.calls[name] = (args, kwargs)
            return result

        return run_phase

    def build_requests(self, samples, *, num_users, requests_per_user, seed):
        return [
            LocomoRequest(user_index, f"sample-{user_index % 2}", f"q{user_index}", "?")
            for user_index in range(num_users)
        ]

    def close_corpus(self) -> None:
        if self.row_maps_when_the_corpus_first_closed is None:
            self.row_maps_when_the_corpus_first_closed = len(self.retrieval)

    def run(self) -> dict[str, Any]:
        return kv_condition.run_kv_injection(self.config, 2)


def test_row_maps_are_dropped_before_the_corpus_they_reference_is_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    kv_run = KVRun(monkeypatch, tmp_path)

    kv_run.run()

    assert kv_run.row_maps_when_the_corpus_first_closed == 0
    assert kv_run.retrieval == {}


@pytest.mark.parametrize("corpus_backend", ["gpu", "cpu"])
def test_composed_memories_stay_in_a_gpu_registry_whatever_the_corpus_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corpus_backend: str
) -> None:
    kv_run = KVRun(monkeypatch, tmp_path, kv_store_backend=corpus_backend)

    kv_run.run()

    namespace = "throughput-kv-phases-cafe"
    assert kv_run.calls["get_gpu_memory_store"] == ((namespace,), {"backend": "gpu"})
    (started_config,), engine_options = kv_run.calls["start_llm"]
    assert started_config is kv_run.config
    assert engine_options == {
        "kv_transfer_config": {
            "kv_connector": "MemoryKVConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "inferscale.v1.kv.connector",
            "kv_connector_extra_config": {
                "memory_namespace": namespace,
                "log_memory_hits": False,
                "memory_store_backend": "gpu",
                "num_staging_slots": 4,
            },
        }
    }
    assert kv_run.calls["drop_namespace"] == ((namespace,), {})


class SamplingParams:
    def __init__(self, extra_args: dict | None = None) -> None:
        self.extra_args = extra_args

    def clone(self) -> SamplingParams:
        return SamplingParams(dict(self.extra_args) if self.extra_args is not None else None)


class MeasuredLLM:
    def __init__(self) -> None:
        self.batches: list[tuple[list, Any]] = []

    def get_tokenizer(self) -> Any:
        return SimpleNamespace(vocab_size=50)

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.batches.append((prompts, sampling_params))
        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[SimpleNamespace(token_ids=[1, 2])],
            )
            for prompt in prompts
        ]

    def reset_prefix_cache(self) -> bool:
        return True


def measure(
    config: ThroughputConfig,
    composed: ComposedRequests,
    *,
    load_stats: dict[str, int] | None = None,
):
    llm = MeasuredLLM()
    connector_events: list[str] = []
    connector = SimpleNamespace(
        reset_load_stats=lambda: connector_events.append(f"reset after {len(llm.batches)} batches"),
        snapshot_load_stats=lambda: (
            connector_events.append(f"snapshot after {len(llm.batches)} batches"),
            load_stats or {"requests_loaded": 2, "requests_covered": 2},
        )[1],
    )
    engine = StartedEngine(llm, SamplingParams(), 0.1)
    measured = kv_condition.measure_kv_batch(
        config, engine, composed, SimpleNamespace(vocab_size=50), connector
    )
    return llm, connector_events, measured


def two_composed_requests() -> ComposedRequests:
    settings: dict[str, Any] = {
        "prompts": [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5, 6, 7]}],
        "memory_user_ids": ["request-00000", "request-00001"],
        "first_memory_user_id": "request-00000",
        "first_memory_token_ids": [1, 2],
    }
    return ComposedRequests(**settings)


def test_load_statistics_cover_only_the_measured_batch(tmp_path: Path) -> None:
    config = make_throughput_config(tmp_path, warmup_batches=2, kv_max_model_len=64)

    llm, connector_events, (measured, load_stats) = measure(config, two_composed_requests())

    assert connector_events == ["reset after 2 batches", "snapshot after 3 batches"]
    assert len(llm.batches) == 3
    assert measured["total_input_tokens"] == 7
    assert load_stats == {"requests_loaded": 2, "requests_covered": 2}


def test_kv_warm_up_prompt_is_routed_like_a_measured_request_and_the_random_ones_are_not(
    tmp_path: Path,
) -> None:
    config = make_throughput_config(tmp_path, warmup_batches=1, kv_max_model_len=64)

    llm, _, _ = measure(config, two_composed_requests())

    (warmup_prompts, warmup_sampling), (measured_prompts, measured_sampling) = llm.batches
    assert [params.extra_args for params in warmup_sampling] == [
        None,
        None,
        {MEMORY_USER_ID_EXTRA_ARG: "request-00000"},
    ]
    assert warmup_prompts[-1]["prompt_token_ids"][:2] == [1, 2]
    assert len(warmup_prompts[-1]["prompt_token_ids"]) > 2
    assert [params.extra_args for params in measured_sampling] == [
        {MEMORY_USER_ID_EXTRA_ARG: "request-00000"},
        {MEMORY_USER_ID_EXTRA_ARG: "request-00001"},
    ]


def test_coverage_counts_distinct_requests_so_reloads_and_native_cache_hits_stay_exact(
    tmp_path: Path,
) -> None:
    config = make_throughput_config(tmp_path, warmup_batches=0, kv_max_model_len=64)
    one_reloaded_one_natively_cached = LoadStats()
    one_reloaded_one_natively_cached.record_load(3, "request-a")
    one_reloaded_one_natively_cached.record_load(3, "request-a")
    one_reloaded_one_natively_cached.record_native_covered("request-b")
    one_reloaded_one_without_memory = LoadStats()
    one_reloaded_one_without_memory.record_load(3, "request-a")
    one_reloaded_one_without_memory.record_load(3, "request-a")

    _, _, (_, load_stats) = measure(
        config, two_composed_requests(), load_stats=one_reloaded_one_natively_cached.snapshot()
    )
    with pytest.raises(RuntimeError, match="KV memory covered 1 of 2 requests"):
        measure(
            config, two_composed_requests(), load_stats=one_reloaded_one_without_memory.snapshot()
        )

    assert (load_stats["requests_loaded"], load_stats["requests_covered"]) == (2, 2)
