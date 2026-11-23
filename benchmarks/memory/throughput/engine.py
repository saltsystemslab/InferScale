from __future__ import annotations

import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from benchmarks.memory.throughput.config import ThroughputConfig
from inferscale.v1.config import EngineConfig
from inferscale.v1.kv.connector_utils import MEMORY_USER_ID_EXTRA_ARG

WARMUP_PROMPT_LIMIT = 10
KV_WARMUP_TAIL_TOKENS = 8


@dataclass(frozen=True, slots=True)
class StartedEngine:
    llm: Any
    sampling_params: Any
    startup_time_s: float


@dataclass(frozen=True, slots=True)
class KVWarmup:
    prompt: dict[str, list[int]]
    sampling_params: Any


def start_llm(
    config: ThroughputConfig,
    *,
    kv_transfer_config: dict[str, Any] | None = None,
    engine: EngineConfig | None = None,
) -> StartedEngine:
    from inferscale.v1.serving.vllm import force_vllm_inprocess_mode, vllm_engine_kwargs

    force_vllm_inprocess_mode()
    from vllm import LLM, SamplingParams

    engine_kwargs = vllm_engine_kwargs(
        model=config.model, dtype=config.kv_dtype, engine=engine or config.engine_config()
    )
    if kv_transfer_config is not None:
        engine_kwargs["kv_transfer_config"] = kv_transfer_config
    started = time.perf_counter()
    llm = LLM(**engine_kwargs)
    startup_time_s = time.perf_counter() - started
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=config.max_output_tokens,
        min_tokens=config.max_output_tokens,
    )
    return StartedEngine(llm, sampling_params, startup_time_s)


def prepare_batch(
    config: ThroughputConfig,
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling_params: Any,
    *,
    kv_warmup: KVWarmup | None = None,
) -> None:
    validate_prompt_lengths(config, prompts)
    warm_up(
        llm, prompts, sampling_params, config.warmup_batches, seed=config.seed, kv_warmup=kv_warmup
    )


def warm_up(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling_params: Any,
    batches: int,
    *,
    seed: int,
    kv_warmup: KVWarmup | None = None,
) -> None:
    if batches > 0 and prompts:
        _generate_warmup_batches(
            llm, prompts, sampling_params, batches, seed=seed, kv_warmup=kv_warmup
        )
    reset_prefix_cache(llm)


def _generate_warmup_batches(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling_params: Any,
    batches: int,
    *,
    seed: int,
    kv_warmup: KVWarmup | None,
) -> None:
    vocab_size = tokenizer_vocab_size(llm.get_tokenizer())
    random_prompts = build_warmup_prompts(prompts, vocab_size=vocab_size, seed=seed)
    warmup_prompts, warmup_sampling = _with_kv_warmup(random_prompts, sampling_params, kv_warmup)
    for _ in range(batches):
        llm.generate(warmup_prompts, warmup_sampling, use_tqdm=False)


def _with_kv_warmup(
    random_prompts: list[dict[str, list[int]]], sampling_params: Any, kv_warmup: KVWarmup | None
) -> tuple[list[dict[str, list[int]]], Any]:
    if kv_warmup is None:
        return random_prompts, sampling_params
    unrouted_sampling = [sampling_params] * len(random_prompts)
    warmup_sampling = [*unrouted_sampling, kv_warmup.sampling_params]
    return [*random_prompts, kv_warmup.prompt], warmup_sampling


def tokenizer_vocab_size(tokenizer: Any) -> int:
    hugging_face_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    return hugging_face_vocab_size or len(tokenizer)


def sampling_params_with_memory_user_ids(
    sampling_params: Any, memory_user_ids: list[str]
) -> list[Any]:
    routed: list[Any] = []
    for memory_user_id in memory_user_ids:
        request_params = _clone_sampling_params(sampling_params)
        extra_args = dict(request_params.extra_args or {})
        extra_args[MEMORY_USER_ID_EXTRA_ARG] = memory_user_id
        request_params.extra_args = extra_args
        routed.append(request_params)
    return routed


def _clone_sampling_params(vllm_sampling_params: Any) -> Any:
    clone = getattr(vllm_sampling_params, "clone", None)
    if not callable(clone):
        raise RuntimeError("vLLM SamplingParams must provide clone() for request routing.")
    return clone()


def build_kv_warmup_prompt(
    memory_token_ids: list[int], *, vocab_size: int, seed: int
) -> dict[str, list[int]]:
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1 to build the KV warmup prompt.")
    rng = random.Random(seed)
    tail = [rng.randrange(vocab_size) for _ in range(KV_WARMUP_TAIL_TOKENS)]
    return {"prompt_token_ids": [*memory_token_ids, *tail]}


def build_warmup_prompts(
    prompts: list[dict[str, list[int]]], *, vocab_size: int, seed: int
) -> list[dict[str, list[int]]]:
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1 to build warmup prompts.")
    rng = random.Random(seed)
    return [
        {"prompt_token_ids": [rng.randrange(vocab_size) for _ in prompt["prompt_token_ids"]]}
        for prompt in prompts[:WARMUP_PROMPT_LIMIT]
    ]


def reset_prefix_cache(llm: Any) -> None:
    reset = getattr(llm, "reset_prefix_cache", None)
    if not callable(reset):
        raise RuntimeError(
            "vLLM LLM has no reset_prefix_cache(); cannot clear the prefix cache "
            "before the measured batch."
        )
    reset_reported_failure = reset() is False
    if reset_reported_failure:
        raise RuntimeError(
            "reset_prefix_cache() reported failure; cached warmup blocks would "
            "contaminate the measured batch."
        )


def measure_batch(
    llm: Any, prompts: list[dict[str, list[int]]], sampling_params: Any
) -> dict[str, int | float]:
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    generation_time_s = time.perf_counter() - started
    if len(outputs) != len(prompts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts.")
    return {
        "generation_time_s": generation_time_s,
        "total_input_tokens": sum(len(output.prompt_token_ids) for output in outputs),
        "total_output_tokens": sum(len(output.outputs[0].token_ids) for output in outputs),
    }


def validate_prompt_lengths(
    config: ThroughputConfig, prompts: Iterable[dict[str, list[int]]]
) -> None:
    longest_prompt_tokens = max((len(prompt["prompt_token_ids"]) for prompt in prompts), default=0)
    required_model_len = longest_prompt_tokens + config.max_output_tokens
    if required_model_len > config.kv_max_model_len:
        raise RuntimeError(
            f"Longest prompt plus output is {required_model_len} tokens, "
            f"exceeding max_model_len={config.kv_max_model_len}."
        )


def release_llm(llm: Any | None) -> None:
    if llm is None:
        return
    from inferscale.v1.serving.vllm import empty_cuda_cache

    del llm
    empty_cuda_cache(collect_ipc=True)
