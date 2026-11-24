from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.throughput import mem0_condition, stores
from benchmarks.memory.throughput.engine import StartedEngine
from benchmarks.memory.throughput.stores import StoreSearch
from benchmarks.memory.throughput.workload import LocomoRequest

SAMPLES = [SimpleNamespace(sample_id=sample_id) for sample_id in ("a", "b", "c")]


def request(user_index: int, sample_id: str = "a") -> LocomoRequest:
    return LocomoRequest(user_index, sample_id, f"q{user_index}", f"query {user_index}")


class Mem0Run:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **settings: Any) -> None:
        self.events: list[Any] = []
        self.config = make_throughput_config(
            tmp_path,
            **{
                "run_id": "mem0-phases",
                "user_counts": (1, 2),
                "requests_per_user": 1,
                **settings,
            },
        )
        llm = SimpleNamespace(get_tokenizer=lambda: "engine tokenizer")
        replacements = {
            "load_samples": lambda config: SAMPLES[:2],
            "fact_catalog_store": lambda config: SimpleNamespace(
                load=lambda sample: (f"fact of {sample.sample_id}",) * 3
            ),
            "build_locomo_requests": lambda samples, *, num_users, requests_per_user, seed: [
                request(user_index, "ab"[user_index % 2]) for user_index in range(num_users)
            ],
            "search_store": self.search,
            "start_llm": lambda config: (
                self.events.append("start engine"),
                StartedEngine(llm, "sampling", 0.5),
            )[1],
            "release_llm": lambda llm: self.events.append("release engine"),
            "extract_memory_scaffold_token_ids": lambda tokenizer, *, block_size: "scaffold",
            "build_memory_prompt_token_ids": self.build_memory,
            "request_prompt_token_ids": lambda tokenizer, memory, locomo_request: [*memory, 9],
            "prepare_batch": lambda config, llm, prompts, sampling_params: self.events.append(
                ("warm up", len(prompts))
            ),
            "measure_batch": self.measure,
        }
        for name, replacement in replacements.items():
            monkeypatch.setattr(mem0_condition, name, replacement)
        monkeypatch.setattr(
            stores,
            "build_user_store",
            lambda config, *, backend, store_root, facts: (
                self.events.append(("build store", backend, store_root.name, facts)),
                f"store of {store_root.name}",
            )[1],
        )
        monkeypatch.setattr(stores, "close_memory", lambda memory: self.events.append("close"))

    def search(self, memory, query, *, top_k):
        self.events.append(("search", memory, query, top_k))
        return StoreSearch(["hit"], 0.5, 0.125)

    def build_memory(self, tokenizer, hits, **options):
        self.events.append(("memory prompt", tokenizer, hits, options))
        return SimpleNamespace(token_ids=[1, 2])

    def measure(self, llm, prompts, sampling_params):
        self.events.append(("measure", prompts, sampling_params))
        return {
            "generation_time_s": 2.0,
            "total_input_tokens": 10 * len(prompts),
            "total_output_tokens": 2 * len(prompts),
        }

    def run(self, backend: str) -> list[dict[str, Any]]:
        return mem0_condition.run_mem0(
            self.config, self.config.user_counts, condition=f"mem0_{backend}", backend=backend
        )


def test_mem0_prompt_build_time_excludes_the_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ticking_clock
) -> None:
    mem0_run = Mem0Run(monkeypatch, tmp_path, user_counts=(1,))
    ticking_clock(mem0_run.events)

    (row,) = mem0_run.run("qdrant")

    steps = [event if isinstance(event, str) else event[0] for event in mem0_run.events]
    search = steps.index("search")
    assert steps[search : search + 4] == ["search", "clock", "memory prompt", "clock"]
    assert row["prompt_build_time_s"] == 7.0
