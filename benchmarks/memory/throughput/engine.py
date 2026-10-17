"""vLLM engine lifecycle, warmup, and batch measurement for the throughput worker."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from benchmarks.memory.throughput.config import ThroughputConfig
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
    """One warmup prompt whose prefix is a registered memory, with its routed sampling params."""

    prompt: dict[str, list[int]]
    sampling_params: Any


def start_llm(
    config: ThroughputConfig, *, kv_transfer_config: dict[str, Any] | None = None
) -> StartedEngine:
    from inferscale.v1.serving.vllm import force_vllm_inprocess_mode, vllm_engine_kwargs

    # Every condition measures the same in-process V1 engine; otherwise QPS
    # differences would be confounded by engine architecture.
    force_vllm_inprocess_mode()
    from vllm import LLM, SamplingParams

    kwargs = vllm_engine_kwargs(
        model=config.model, dtype=config.kv_dtype, engine=config.engine_config()
    )
    if kv_transfer_config is not None:
        kwargs["kv_transfer_config"] = kv_transfer_config
    started = time.perf_counter()
    llm = LLM(**kwargs)
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
    """Check the batch fits the engine and warm the engine up for it."""
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
    """Warm up with random tokens so no real prompt lands in the prefix cache.

    Prefix caching is enabled, so warming up with the measured prompts would
    let the measured batch hit the cache (inflated QPS) and, worse, let the
    native cache satisfy memory prefixes so the connector never injects KV.
    Warmup only needs to exercise kernels, allocator, and scheduler; token
    identity is irrelevant. ``kv_warmup`` carries explicit memory routing so
    the injection path warms too.
    """
    if batches > 0 and prompts:
        vocab_size = tokenizer_vocab_size(llm.get_tokenizer())
        warmup_prompts = build_warmup_prompts(prompts, vocab_size=vocab_size, seed=seed)
        warmup_sampling: Any = sampling_params
        if kv_warmup is not None:
            warmup_sampling = [
                *([sampling_params] * len(warmup_prompts)),
                kv_warmup.sampling_params,
            ]
            warmup_prompts = [*warmup_prompts, kv_warmup.prompt]
        for _ in range(batches):
            llm.generate(warmup_prompts, warmup_sampling, use_tqdm=False)
    # Reset even with warmup disabled: conditions that measure several user
    # counts against one engine must not let count N's prompts serve count
    # N+1 from the native cache.
    reset_prefix_cache(llm)


def tokenizer_vocab_size(tokenizer: Any) -> int:
    # Hugging Face tokenizers expose vocab_size; other tokenizers only their length.
    return int(getattr(tokenizer, "vocab_size", 0) or 0) or len(tokenizer)


def sampling_params_with_memory_user_ids(
    sampling_params: Any, memory_user_ids: list[str]
) -> list[Any]:
    """Clone the base params once per request, routing each to its memory."""
    # SamplingParams is vLLM's type; the benchmark only relies on its clone().
    clone = getattr(sampling_params, "clone", None)
    if not callable(clone):
        raise RuntimeError("vLLM SamplingParams must provide clone() for request routing.")
    routed: list[Any] = []
    for memory_user_id in memory_user_ids:
        request_params = clone()
        extra_args = dict(request_params.extra_args or {})
        extra_args[MEMORY_USER_ID_EXTRA_ARG] = memory_user_id
        request_params.extra_args = extra_args
        routed.append(request_params)
    return routed


def build_kv_warmup_prompt(
    memory_token_ids: list[int], *, vocab_size: int, seed: int
) -> dict[str, list[int]]:
    """One warmup prompt whose prefix is a registered memory.

    Random warmup prompts match no memory, so without this the connector's
    match and inject path (scatter kernels, staging) would first run inside
    the measured batch. The prompt must be strictly longer than the memory
    for the match to fire; the tail is seeded-random so it shares nothing
    with real queries.
    """
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
    # Fail closed: measurement isolation depends on this reset (the KV warmup
    # prompt shares blocks with a measured request's memory, and prompts
    # repeat across user counts), so a stale cache must abort, not warn.
    reset = getattr(llm, "reset_prefix_cache", None)
    if not callable(reset):
        raise RuntimeError(
            "vLLM LLM has no reset_prefix_cache(); cannot clear the prefix cache "
            "before the measured batch."
        )
    if reset() is False:
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
    longest = max((len(prompt["prompt_token_ids"]) for prompt in prompts), default=0)
    required = longest + config.max_output_tokens
    if required > config.kv_max_model_len:
        raise RuntimeError(
            f"Longest prompt plus output is {required} tokens, "
            f"exceeding max_model_len={config.kv_max_model_len}."
        )


def release_llm(llm: Any | None) -> None:
    if llm is None:
        return
    from inferscale.v1.serving.vllm import empty_cuda_cache

    del llm
    empty_cuda_cache(collect_ipc=True)
