from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from library_fakes import FakeTokenizer
from throughput_config import make_throughput_config

from benchmarks.common.lmcache.runtime import WarmupStats
from benchmarks.common.lmcache.segments import separator_token_ids
from benchmarks.common.vector_types import SearchHit
from benchmarks.memory.prompting import extract_memory_scaffold_token_ids
from benchmarks.memory.throughput import lmcache_condition, stores
from benchmarks.memory.throughput.engine import StartedEngine
from benchmarks.memory.throughput.lmcache_condition import (
    catalog_warm_up_prompts,
    prime_fact_cache,
    run_lmcache_blend,
)
from benchmarks.memory.throughput.stores import StoreSearch
from benchmarks.memory.throughput.workload import LocomoRequest

BYTES_PER_TOKEN = 1000
CACHE_USAGE_BYTES = 123_456


def fact(memory_id: str, text: str, rank: int) -> SearchHit:
    payload = {"memory": text, "source_session_index": 0, "source_turn_index": rank}
    return SearchHit(id=memory_id, payload=payload, score=1.0, distance=0.0, rank=rank)


CATALOG = [fact("m1", "Alice likes tea.", 1), fact("m2", "Bob plays chess.", 2)]


def tokenizer_with_vocabulary() -> FakeTokenizer:
    tokenizer = FakeTokenizer()
    tokenizer.vocab_size = 300
    return tokenizer


class BlendRun:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        matched_tokens_per_request: int = 10**6,
        **settings: Any,
    ) -> None:
        self.events: list[Any] = []
        self.measured_batches: list[list[dict[str, list[int]]]] = []
        self.tokenizer = tokenizer_with_vocabulary()
        self.separator = separator_token_ids(self.tokenizer, " # # ")
        self.config = make_throughput_config(
            tmp_path,
            **{
                "run_id": "blend-phases",
                "user_counts": (1, 2),
                "requests_per_user": 1,
                "warmup_batches": 1,
                "embedding_api_key": "test-key",
                "lmcache_max_local_cpu_size_gb": 1.0,
                **settings,
            },
        )
        llm = SimpleNamespace(
            get_tokenizer=lambda: self.tokenizer,
            generate=lambda prompts, sampling_params, use_tqdm=False: self.events.append(
                "random warm-up"
            ),
            reset_prefix_cache=lambda: True,
        )
        requests = [
            LocomoRequest(user_index, "sample-0", f"q{user_index}", f"query {user_index}")
            for user_index in range(2)
        ]

        def measure(llm, prompts, sampling_params):
            self.events.append("measure")
            self.measured_batches.append(prompts)
            return {
                "generation_time_s": 2.0,
                "total_input_tokens": 40 * len(prompts),
                "total_output_tokens": 50 * len(prompts),
            }

        def snapshot_load_stats():
            request_count = len(self.measured_batches[-1])
            return {
                "tokens_matched": 100 * request_count,
                "load_time_ms": 4.0 * request_count,
                "matched_tokens": {
                    str(index): matched_tokens_per_request for index in range(request_count)
                },
            }

        def warm_prompts(llm, prompts, *, batch_size):
            self.events.append("fact warm-up")
            return WarmupStats(
                prompt_count=len(prompts), token_count=sum(map(len, prompts)), time_ms=500.0
            )

        replacements = {
            "load_samples": lambda config: [SimpleNamespace(sample_id="sample-0")],
            "fact_catalog_store": lambda config: SimpleNamespace(load=lambda sample: CATALOG),
            "fact_catalog_hits": list,
            "build_locomo_requests": lambda *args, **kwargs: requests,
            "search_store": lambda memory, query, *, top_k: StoreSearch(CATALOG, 0.5, 0.125),
            "start_llm": lambda config, **options: StartedEngine(llm, object(), 0.25),
            "release_llm": lambda llm: self.events.append("release_llm"),
            "register_model": lambda llm: None,
            "lmcache_separator": lambda: self.separator,
            "kv_bytes_per_token": lambda: BYTES_PER_TOKEN,
            "clear_cache": lambda: self.events.append("clear cache"),
            "cpu_cache_usage_bytes": lambda: CACHE_USAGE_BYTES,
            "warm_prompts": warm_prompts,
            "reset_load_stats": lambda: self.events.append("reset load stats"),
            "snapshot_load_stats": snapshot_load_stats,
            "measure_batch": measure,
            "shutdown": lambda: self.events.append("shutdown"),
        }
        for name, replacement in replacements.items():
            monkeypatch.setattr(lmcache_condition, name, replacement)
        monkeypatch.setattr(stores, "build_user_store", lambda config, **store: object())
        monkeypatch.setattr(stores, "close_memory", lambda memory: self.events.append("close"))

    def run(self) -> list[dict[str, Any]]:
        return run_lmcache_blend(self.config, self.config.user_counts)


def test_random_warm_up_precedes_the_fact_warm_up_so_eviction_drops_its_junk_before_the_facts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blend_run = BlendRun(monkeypatch, tmp_path, user_counts=(2,))

    blend_run.run()

    assert blend_run.events.index("random warm-up") < blend_run.events.index("fact warm-up")


def test_batch_whose_requests_were_partly_recomputed_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blend_run = BlendRun(monkeypatch, tmp_path, matched_tokens_per_request=3)

    with pytest.raises(RuntimeError, match="for 0 of 1 requests .worst request matched 3 of"):
        blend_run.run()

    assert blend_run.events[-3:] == ["close", "shutdown", "release_llm"]


def test_priming_fails_when_the_cache_holds_less_than_half_of_the_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lmcache_condition, "clear_cache", lambda: None)
    monkeypatch.setattr(
        lmcache_condition,
        "warm_prompts",
        lambda llm, prompts, *, batch_size: WarmupStats(2, 5, 1500.0),
    )
    monkeypatch.setattr(lmcache_condition, "cpu_cache_usage_bytes", lambda: 24)

    with pytest.raises(RuntimeError) as failure:
        prime_fact_cache(object(), [[1, 2, 3], [4, 5]], bytes_per_token=10)

    assert str(failure.value) == (
        "LMCache holds 24 bytes after the fact warm-up, below half of the "
        "expected 50; the fact segments were not stored. Check the separator "
        "parity and the LMCache logs."
    )


def test_warm_up_prompts_are_the_header_and_each_distinct_fact_of_every_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lmcache_condition, "fact_catalog_hits", list)
    tokenizer = tokenizer_with_vocabulary()
    scaffold = extract_memory_scaffold_token_ids(tokenizer, block_size=16)
    separator = separator_token_ids(tokenizer, " # # ")
    other_catalog = [fact("m3", "Cara runs.", 1), CATALOG[0]]

    prompts = catalog_warm_up_prompts(tokenizer, scaffold, [CATALOG, other_catalog], separator)

    def fact_tokens(text: str) -> list[int]:
        return tokenizer.encode(text + "\n", add_special_tokens=False)

    assert prompts == [
        scaffold.header_token_ids,
        fact_tokens("Bob plays chess."),
        fact_tokens("Alice likes tea."),
        fact_tokens("Cara runs."),
    ]
