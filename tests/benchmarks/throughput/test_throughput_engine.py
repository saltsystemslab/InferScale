from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.throughput import kv_condition, lmcache_condition, mem0_condition
from benchmarks.memory.throughput.engine import (
    measure_batch,
    start_llm,
    tokenizer_vocab_size,
    warm_up,
)

MULTIPROCESSING_VARIABLE = "VLLM_ENABLE_V1_MULTIPROCESSING"


class FakeVllm(types.ModuleType):
    def __init__(self, events: list[Any]) -> None:
        super().__init__("vllm")
        self.events = events

    def __getattr__(self, name: str) -> Any:
        if name not in {"LLM", "SamplingParams"}:
            raise AttributeError(name)
        self.events.append((f"import {name}", os.environ.get(MULTIPROCESSING_VARIABLE)))
        return lambda **kwargs: self.construct(name, kwargs)

    def construct(self, name: str, kwargs: dict[str, Any]) -> Any:
        self.events.append((name, kwargs))
        return SimpleNamespace(kind=name, kwargs=kwargs)


@pytest.fixture
def vllm_events(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    events: list[Any] = []
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setenv(MULTIPROCESSING_VARIABLE, "1")
    monkeypatch.setitem(sys.modules, "vllm", FakeVllm(events))
    return events


def test_every_condition_measures_the_same_in_process_engine_so_qps_is_not_an_engine_difference(
    tmp_path: Path, vllm_events: list[Any]
) -> None:
    start_llm(make_throughput_config(tmp_path))

    multiprocessing_when_vllm_loaded = {
        setting for step, setting in vllm_events if step.startswith("import ")
    }
    assert multiprocessing_when_vllm_loaded == {"0"}
    assert mem0_condition.start_llm is start_llm
    assert lmcache_condition.start_llm is start_llm
    assert kv_condition.start_llm is start_llm


def test_engine_starts_in_process_with_fixed_length_deterministic_sampling(
    tmp_path: Path, vllm_events: list[Any]
) -> None:
    config = make_throughput_config(
        tmp_path,
        model="org/answer-model",
        max_output_tokens=7,
        kv_dtype="float16",
        kv_gpu_memory_utilization=0.6,
        kv_max_model_len=4096,
        kv_block_size=32,
        kv_enable_prefix_caching=False,
    )

    started = start_llm(config)

    assert vllm_events == [
        ("import LLM", "0"),
        ("import SamplingParams", "0"),
        (
            "LLM",
            {
                "model": "org/answer-model",
                "dtype": "float16",
                "trust_remote_code": True,
                "enable_prefix_caching": False,
                "disable_log_stats": False,
                "swap_space": 0,
                "cpu_offload_gb": 0,
                "gpu_memory_utilization": 0.6,
                "block_size": 32,
                "max_model_len": 4096,
            },
        ),
        ("SamplingParams", {"temperature": 0.0, "top_p": 1.0, "max_tokens": 7, "min_tokens": 7}),
    ]
    assert started.llm.kind == "LLM"
    assert started.sampling_params.kind == "SamplingParams"
    assert started.startup_time_s >= 0.0


def test_engine_startup_time_covers_only_the_engine_construction(
    tmp_path: Path, vllm_events: list[Any], ticking_clock
) -> None:
    ticking_clock(vllm_events)

    started = start_llm(make_throughput_config(tmp_path))

    assert [event if event == "clock" else event[0] for event in vllm_events] == [
        "import LLM",
        "import SamplingParams",
        "clock",
        "LLM",
        "clock",
        "SamplingParams",
    ]
    assert started.startup_time_s == 3.0


class RecordingLLM:
    def __init__(self, events: list[Any], *, reset_result: Any = True):
        self.events = events
        self.reset_result = reset_result

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.events.append(("generate", prompts, sampling_params, use_tqdm))
        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[SimpleNamespace(token_ids=[7, 8, 9])],
            )
            for prompt in prompts
        ]

    def reset_prefix_cache(self) -> Any:
        self.events.append(("reset_prefix_cache",))
        return self.reset_result


def prompts_of_lengths(lengths: list[int]) -> list[dict[str, list[int]]]:
    return [{"prompt_token_ids": [1000 + index] * length} for index, length in enumerate(lengths)]


@pytest.mark.parametrize("batches,prompt_lengths", [(0, [4]), (2, [])])
def test_cache_is_reset_even_without_warm_up_so_one_user_count_cannot_serve_the_next(
    batches: int, prompt_lengths: list[int]
) -> None:
    events: list[Any] = []

    warm_up(RecordingLLM(events), prompts_of_lengths(prompt_lengths), object(), batches, seed=1)

    assert events == [("reset_prefix_cache",)]


def test_failed_cache_reset_aborts_because_stale_blocks_would_serve_the_measured_batch() -> None:
    with pytest.raises(RuntimeError) as failure:
        warm_up(RecordingLLM([], reset_result=False), [], object(), 1, seed=1)

    assert str(failure.value) == (
        "reset_prefix_cache() reported failure; cached warmup blocks would "
        "contaminate the measured batch."
    )


class TokenizerWithLength:
    def __init__(self, length: int, **attributes: Any) -> None:
        self.length = length
        vars(self).update(attributes)

    def __len__(self) -> int:
        return self.length


@pytest.mark.parametrize(
    "tokenizer,expected",
    [
        (TokenizerWithLength(7, vocab_size=32000), 32000),
        (TokenizerWithLength(7), 7),
    ],
)
def test_vocab_size_comes_from_the_tokenizer_attribute_or_else_its_length(
    tokenizer: Any, expected: int
) -> None:
    assert tokenizer_vocab_size(tokenizer) == expected


def test_measured_batch_reports_generation_time_and_token_totals(ticking_clock) -> None:
    events: list[Any] = []
    ticking_clock(events)
    prompts = prompts_of_lengths([3, 5, 2])
    sampling_params = object()

    measured = measure_batch(RecordingLLM(events), prompts, sampling_params)

    assert events == ["clock", ("generate", prompts, sampling_params, False), "clock"]
    assert measured == {
        "generation_time_s": 3.0,
        "total_input_tokens": 10,
        "total_output_tokens": 9,
    }
    assert list(measured) == ["generation_time_s", "total_input_tokens", "total_output_tokens"]
