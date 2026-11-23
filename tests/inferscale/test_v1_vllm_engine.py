from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import inferscale.v1.serving.vllm as vllm_module
from inferscale.v1.config import EngineConfig
from inferscale.v1.serving.vllm import (
    VLLM_CLI_ENV_VARIABLES,
    VLLMEngine,
    force_vllm_inprocess_mode,
)

MULTIPROCESSING = "VLLM_ENABLE_V1_MULTIPROCESSING"
BUILD_SAMPLING_S = 1.0
GENERATE_S = 4.0
ENGINE_CONFIG = EngineConfig(block_size=8, max_model_len=512, gpu_memory_utilization=0.3)


class ManualClock:
    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeVLLM:
    def __init__(self, clock: ManualClock) -> None:
        self.clock = clock
        self.sampled_with: list[dict[str, Any]] = []
        self.generations: list[tuple[Any, Any, bool]] = []
        self.tokenizer = object()

    def LLM(self, **settings: Any) -> FakeVLLM:
        return self

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def SamplingParams(self, **settings: Any) -> dict[str, Any]:
        self.clock.advance(BUILD_SAMPLING_S)
        self.sampled_with.append(settings)
        return settings

    def generate(self, prompts: Any, sampling: Any, use_tqdm: bool) -> list[SimpleNamespace]:
        self.clock.advance(GENERATE_S)
        self.generations.append((prompts, sampling, use_tqdm))
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text="  the answer \n")],
                metrics=SimpleNamespace(first_token_latency=0.0625),
            )
        ]


class ImportedVLLM:
    def __init__(self, vllm: FakeVLLM) -> None:
        self.vllm = vllm
        self.environment_at_import: dict[str, str | None] = {}

    @property
    def LLM(self) -> Any:
        self.environment_at_import = {
            name: os.environ.get(name) for name in (*VLLM_CLI_ENV_VARIABLES, MULTIPROCESSING)
        }
        return self.vllm.LLM

    @property
    def SamplingParams(self) -> Any:
        return self.vllm.SamplingParams


@pytest.fixture
def shell(monkeypatch) -> pytest.MonkeyPatch:
    for name in (*VLLM_CLI_ENV_VARIABLES, MULTIPROCESSING):
        monkeypatch.setenv(name, "restored after the test")
        monkeypatch.delenv(name)
    return monkeypatch


@pytest.fixture
def clock(monkeypatch) -> ManualClock:
    clock = ManualClock()
    monkeypatch.setattr(vllm_module, "time", clock)
    return clock


@pytest.fixture
def vllm(shell, clock) -> FakeVLLM:
    vllm = FakeVLLM(clock)
    shell.setitem(sys.modules, "vllm", vllm)
    return vllm


@pytest.fixture
def engine(vllm) -> VLLMEngine:
    return VLLMEngine("model/name", "bfloat16", ENGINE_CONFIG)


def test_cli_shell_variables_are_cleared_before_vllm_loads_and_rejects_unknown_settings(
    shell, clock
) -> None:
    for name in VLLM_CLI_ENV_VARIABLES:
        shell.setenv(name, "set by the shell")
    shell.setenv(MULTIPROCESSING, "1")
    imported = ImportedVLLM(FakeVLLM(clock))
    shell.setitem(sys.modules, "vllm", imported)

    VLLMEngine("model/name", "bfloat16", ENGINE_CONFIG).start()

    assert imported.environment_at_import == {
        **dict.fromkeys(VLLM_CLI_ENV_VARIABLES),
        MULTIPROCESSING: "0",
    }


def test_environment_vllm_already_read_is_switched_to_in_process_mode_too(shell) -> None:
    loaded_envs = SimpleNamespace(VLLM_ENABLE_V1_MULTIPROCESSING=True)
    shell.setitem(sys.modules, "vllm.envs", loaded_envs)

    force_vllm_inprocess_mode()

    assert loaded_envs.VLLM_ENABLE_V1_MULTIPROCESSING is False
    assert os.environ[MULTIPROCESSING] == "0"


def test_generate_reports_stripped_text_engine_ttft_and_the_time_around_generation(
    engine, vllm, clock
) -> None:
    engine.start()
    before_generate = clock.now

    output = engine.generate((5, 6, 7), max_tokens=16, temperature=0.25, top_p=0.5)

    assert vllm.sampled_with == [{"temperature": 0.25, "top_p": 0.5, "max_tokens": 16}]
    assert vllm.generations == [([{"prompt_token_ids": [5, 6, 7]}], vllm.sampled_with[0], False)]
    assert output.text == "the answer"
    assert output.ttft_ms == 62.5
    assert output.started_at == before_generate + BUILD_SAMPLING_S
    assert output.finished_at == output.started_at + GENERATE_S
