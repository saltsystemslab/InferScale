from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from library_fakes import FakeTokenizer
from throughput_config import make_throughput_config

from benchmarks.common.config import ConfigError
from benchmarks.common.lmcache.runtime import WarmupStats
from benchmarks.common.lmcache.segments import contains_separator, separator_token_ids
from benchmarks.common.vector_types import SearchHit
from benchmarks.memory.prompting import (
    build_memory_prompt_token_ids,
    extract_memory_scaffold_token_ids,
)
from benchmarks.memory.throughput import engine, lmcache_condition, stores
from benchmarks.memory.throughput.engine import StartedEngine
from benchmarks.memory.throughput.lmcache_condition import (
    count_covered_requests,
    require_fact_cache_resident,
    require_full_coverage,
    run_lmcache_blend,
)
from benchmarks.memory.throughput.stores import StoreSearch
from benchmarks.memory.throughput.workload import LocomoRequest, request_prompt_token_ids
from inferscale.v1.units import bytes_to_mb

SEPARATOR_STR = " # # "
BYTES_PER_TOKEN = 1000
CACHE_USAGE_BYTES = 123_456


def _hit(memory_id: str, text: str, rank: int) -> SearchHit:
    payload = {"memory": text, "source_session_index": 0, "source_turn_index": rank}
    return SearchHit(id=memory_id, payload=payload, score=1.0, distance=0.0, rank=rank)


CATALOG = [_hit("m1", "Alice likes tea.", 1), _hit("m2", "Bob plays chess.", 2)]


class _RecordingLLM:
    """Records every engine call the condition makes, in order."""

    def __init__(self, events: list) -> None:
        self.events = events
        self.tokenizer = FakeTokenizer()
        self.tokenizer.vocab_size = 300  # the random warm-up draws tokens below this

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.events.append(("generate", len(prompts)))

    def reset_prefix_cache(self):
        self.events.append(("reset_prefix_cache",))
        return True


class _Harness:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides) -> None:
        self.events: list = []
        self.llm = _RecordingLLM(self.events)
        self.started: list[dict] = []
        self.measured: list[list[dict[str, list[int]]]] = []
        settings = {
            "run_id": "lmcache-test",
            "user_counts": (1, 2),
            "requests_per_user": 1,
            "warmup_batches": 1,
            "embedding_api_key": "test-key",
            "lmcache_max_local_cpu_size_gb": 1.0,
            **overrides,
        }
        self.config = make_throughput_config(tmp_path, **settings)
        samples = [SimpleNamespace(sample_id="sample-0", qa=[])]
        requests = [
            LocomoRequest(
                user_index=index,
                sample_id="sample-0",
                question_id=f"q{index}",
                query=f"query {index}",
            )
            for index in range(2)
        ]
        stores_built: dict[str, SimpleNamespace] = {}

        def build_store(config, *, backend, store_root, facts):
            assert backend == "qdrant"
            store = SimpleNamespace(name=store_root.name)
            stores_built[store_root.name] = store
            return store

        def search(memory, query, *, top_k):
            assert memory is stores_built["sample-0"]
            return StoreSearch(list(CATALOG), 0.02, 0.003)

        def start_llm(config, *, kv_transfer_config=None, engine=None):
            self.started.append({"kv_transfer_config": kv_transfer_config, "engine": engine})
            self.events.append(("start_llm",))
            return StartedEngine(self.llm, SimpleNamespace(max_tokens=50), 0.25)

        def warm(llm, prompts, *, batch_size):
            self.events.append(("warm_prompts", [list(prompt) for prompt in prompts]))
            return WarmupStats(
                prompt_count=len(prompts), token_count=sum(map(len, prompts)), time_ms=500.0
            )

        def measure(llm, prompts, sampling_params):
            self.events.append(("measure", len(prompts)))
            self.measured.append(prompts)
            return {
                "generation_time_s": 2.0,
                "total_input_tokens": 40 * len(prompts),
                "total_output_tokens": 50 * len(prompts),
            }

        def snapshot():
            self.events.append(("snapshot_load_stats",))
            count = len(self.measured[-1])
            return {
                "requests_seen": count,
                "requests_matched": count,
                "tokens_matched": 100 * count,
                "load_calls": count,
                "load_time_ms": 4.0 * count,
                "matched_tokens": {str(index): 10**6 for index in range(count)},
            }

        patches = {
            "load_samples": lambda config: samples,
            "fact_catalog_store": lambda config: SimpleNamespace(load=lambda sample: list(CATALOG)),
            "fact_catalog_hits": lambda facts: list(facts),
            "build_locomo_requests": lambda *args, **kwargs: requests,
            "search_store": search,
            "start_llm": start_llm,
            "release_llm": lambda llm: self.events.append(("release_llm",)),
            "register_model": lambda llm: self.events.append(("register_model", llm)),
            "require_separator_parity": lambda tokenizer, text: self.events.append(("parity", text))
            or separator_token_ids(tokenizer, text),
            "kv_bytes_per_token": lambda: BYTES_PER_TOKEN,
            "clear_cache": lambda: self.events.append(("clear_cache",)) or 0,
            "cpu_cache_usage_bytes": lambda: self.events.append(("cpu_cache_usage_bytes",))
            or CACHE_USAGE_BYTES,
            "warm_prompts": warm,
            "reset_load_stats": lambda: self.events.append(("reset_load_stats",)),
            "snapshot_load_stats": snapshot,
            "measure_batch": measure,
        }
        for name, value in patches.items():
            monkeypatch.setattr(lmcache_condition, name, value)
        monkeypatch.setattr(stores, "build_user_store", build_store)
        monkeypatch.setattr(
            stores, "close_memory", lambda memory: self.events.append(("close", memory.name))
        )
        monkeypatch.setattr(engine, "validate_prompt_lengths", lambda *args: None)


def test_run_lmcache_blend_orders_phases_and_maps_the_result_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)

    rows = run_lmcache_blend(harness.config, harness.config.user_counts)

    (started,) = harness.started
    assert started["kv_transfer_config"]["kv_connector"] == "BlendConnector"
    assert (
        started["kv_transfer_config"]["kv_connector_extra_config"]["lmcache.max_local_cpu_size"]
        == 1.0
    )
    assert started["engine"].enable_prefix_caching is False
    kinds = [event[0] for event in harness.events]
    assert kinds[:3] == ["start_llm", "register_model", "parity"]
    per_count = [
        "generate",  # the random-token warm-up
        "reset_prefix_cache",
        "clear_cache",
        "warm_prompts",
        "cpu_cache_usage_bytes",
        "reset_load_stats",
        "measure",
        "snapshot_load_stats",
    ]
    assert kinds[3:] == per_count * 2 + ["close", "release_llm"]
    warm_events = [event for event in harness.events if event[0] == "warm_prompts"]
    scaffold = extract_memory_scaffold_token_ids(
        harness.llm.tokenizer, block_size=harness.config.kv_block_size
    )
    assert warm_events[0][1][0] == scaffold.header_token_ids
    assert len(warm_events[0][1]) == 1 + len(CATALOG)
    assert [event[1] for event in harness.events if event[0] == "measure"] == [1, 2]

    one, two = rows
    assert (one["condition"], one["vector_backend"]) == ("lmcache_blend", "qdrant")
    assert one["kv_store_backend"] == "lmcache-cpu"
    assert one["kv_prefix_caching"] == 0
    assert one["kv_store_host_mb"] == bytes_to_mb(CACHE_USAGE_BYTES)
    assert one["kv_precompute_time_s"] == 0.5
    assert one["kv_requests_loaded"] == 1 and two["kv_requests_loaded"] == 2
    assert two["kv_h2d_bytes"] == 200 * BYTES_PER_TOKEN
    assert two["kv_h2d_avg_ms"] == 4.0
    assert one["fact_count"] == len(CATALOG)
    assert two["throughput_qps"] == 1.0
    assert two["retrieval_time_s"] == pytest.approx(0.04)


def test_blend_prompt_is_the_mem0_prompt_plus_separators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path)
    run_lmcache_blend(harness.config, harness.config.user_counts)
    tokenizer = harness.llm.tokenizer
    separator = separator_token_ids(tokenizer, SEPARATOR_STR)
    scaffold = extract_memory_scaffold_token_ids(tokenizer, block_size=harness.config.kv_block_size)
    request = LocomoRequest(user_index=0, sample_id="sample-0", question_id="q0", query="query 0")
    mem0_memory = build_memory_prompt_token_ids(
        tokenizer,
        SimpleNamespace(sample_id="sample-0"),
        CATALOG,
        context_window=0,
        memory_scaffold=scaffold,
    ).token_ids
    mem0_prompt = request_prompt_token_ids(tokenizer, mem0_memory, request)

    (blend_prompt,) = [prompt["prompt_token_ids"] for prompt in harness.measured[0]]

    assert contains_separator(blend_prompt, separator)
    assert len(blend_prompt) == len(mem0_prompt) + (len(CATALOG) + 1) * len(separator)
    assert _without(blend_prompt, separator) == mem0_prompt


def _without(token_ids: list[int], separator: list[int]) -> list[int]:
    kept: list[int] = []
    index = 0
    while index < len(token_ids):
        if token_ids[index : index + len(separator)] == separator:
            index += len(separator)
        else:
            kept.append(token_ids[index])
            index += 1
    return kept


def test_run_lmcache_blend_requires_the_lmcache_settings_before_starting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = make_throughput_config(tmp_path, embedding_api_key="test-key")
    monkeypatch.setattr(
        lmcache_condition, "start_llm", lambda *args, **kwargs: pytest.fail("engine started")
    )

    with pytest.raises(ConfigError, match="max_local_cpu_size_gb is required"):
        run_lmcache_blend(config, (2,))


def test_count_covered_requests_orders_by_vllm_request_id() -> None:
    assert count_covered_requests({"10": 100, "9": 50}, [50, 100]) == 2
    assert count_covered_requests({"9": 49, "10": 100}, [50, 100]) == 1
    with pytest.raises(RuntimeError, match="recorded 1 requests for a batch of 2"):
        count_covered_requests({"9": 49}, [50, 100])


def test_incomplete_coverage_names_the_worst_request() -> None:
    with pytest.raises(RuntimeError, match=r"1 of 2 requests \(worst request matched 49 of 50"):
        require_full_coverage(1, 2, {"0": 49, "1": 100}, [50, 100])
    require_full_coverage(2, 2, {"0": 50, "1": 100}, [50, 100])


def test_fact_cache_residency_check() -> None:
    require_fact_cache_resident(None, expected_bytes=10)
    require_fact_cache_resident(5, expected_bytes=10)
    with pytest.raises(RuntimeError, match="were not stored"):
        require_fact_cache_resident(4, expected_bytes=10)


def test_capacity_check_fails_before_the_warm_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    harness = _Harness(monkeypatch, tmp_path, lmcache_max_local_cpu_size_gb=1e-6)

    with pytest.raises(RuntimeError, match="lmcache.max_local_cpu_size_gb"):
        run_lmcache_blend(harness.config, harness.config.user_counts)

    assert "warm_prompts" not in [event[0] for event in harness.events]
