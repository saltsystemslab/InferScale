from __future__ import annotations

import argparse
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from ..results import write_json
from ..vector_types import VectorStoreConfig
from .config import ALL_CONDITIONS, BenchmarkPoint, ThroughputConfig, parse_matrix
from .reporting import RESULT_COLUMNS
from .synthetic import (
    build_memory_prompt,
    build_no_memory_prompt,
    build_requests,
    build_retrieval_prompt,
    build_synthetic_memory,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Internal throughput benchmark worker.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=ALL_CONDITIONS, required=True)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = ThroughputConfig.from_json_file(args.config)
    matrix = parse_matrix(args.matrix)
    unknown = [point for point in matrix if point not in config.matrix]
    if unknown:
        parser.error("Worker matrix is not in the configured matrix: " + ", ".join(map(str, unknown)))

    results = run_condition(config, args.condition, matrix)
    write_json(args.output, {"condition": args.condition, "results": results})
    print(f"worker wrote {len(results)} row(s) to {args.output}", flush=True)


def run_condition(
    config: ThroughputConfig,
    condition: str,
    matrix: tuple[BenchmarkPoint, ...],
) -> list[dict[str, Any]]:
    if condition == "no_memory":
        return _run_generation_condition(config, matrix, include_memory=False)
    if condition == "prompt_injection":
        return _run_generation_condition(config, matrix, include_memory=True)
    if condition == "kv_injection":
        if len(matrix) != 1:
            raise ValueError("A KV worker must receive exactly one matrix point.")
        return [_run_kv_injection(config, matrix[0])]
    if condition == "mem0":
        return _run_mem0(config, matrix)
    raise ValueError(f"Unsupported condition: {condition}")


def _run_generation_condition(
    config: ThroughputConfig,
    matrix: tuple[BenchmarkPoint, ...],
    *,
    include_memory: bool,
) -> list[dict[str, Any]]:
    condition = "prompt_injection" if include_memory else "no_memory"
    llm: Any | None = None
    try:
        llm, sampling_params, engine_startup_time_s = _start_llm(config)
        tokenizer = llm.get_tokenizer()
        results: list[dict[str, Any]] = []
        for point in matrix:
            print(f"{condition}: users={point.num_users} memory_tokens={point.memory_tokens}", flush=True)
            prompt_started = time.perf_counter()
            memories = {}
            if include_memory:
                memories = {
                    user_index: build_synthetic_memory(
                        tokenizer,
                        user_index=user_index,
                        memory_tokens=point.memory_tokens,
                        seed=config.seed,
                    )
                    for user_index in range(point.num_users)
                }
            prompts = []
            for request in build_requests(
                num_users=point.num_users,
                requests_per_user=config.requests_per_user,
                seed=config.seed,
            ):
                if include_memory:
                    token_ids = build_memory_prompt(
                        tokenizer,
                        memories[request.user_index].memory_token_ids,
                        request.query,
                    )
                else:
                    token_ids = build_no_memory_prompt(tokenizer, request.query)
                prompts.append({"prompt_token_ids": token_ids})
            prompt_build_time_s = time.perf_counter() - prompt_started
            _validate_prompt_lengths(config, prompts)
            _warm_up(llm, prompts, sampling_params, config.warmup_batches)
            measured = _measure_batch(llm, prompts, sampling_params)
            results.append(
                _result_row(
                    config,
                    point,
                    condition=condition,
                    wall_time_s=measured["generation_time_s"],
                    generation_time_s=measured["generation_time_s"],
                    prompt_build_time_s=prompt_build_time_s,
                    engine_startup_time_s=engine_startup_time_s,
                    total_input_tokens=measured["total_input_tokens"],
                    total_output_tokens=measured["total_output_tokens"],
                )
            )
        return results
    finally:
        _release_llm(llm)


def _run_kv_injection(config: ThroughputConfig, point: BenchmarkPoint) -> dict[str, Any]:
    if config.kv_device not in {"cuda", "cuda:0"}:
        raise RuntimeError("The current strict GPU registry requires --device cuda:0.")

    from ..kv.chunked_rope import ChunkedRopeEncoder
    from ..kv.gpu_registry import drop_namespace, namespace_stats, register_user_memory
    from ..kv.vllm_runtime import build_strict_gpu_kv_transfer_config, force_vllm_inprocess_mode

    force_vllm_inprocess_mode()
    namespace = f"throughput-{config.run_id}-{uuid.uuid4().hex}"
    encoder: Any | None = None
    llm: Any | None = None
    try:
        print(
            f"kv_injection: users={point.num_users} memory_tokens={point.memory_tokens} precompute starting",
            flush=True,
        )
        precompute_started = time.perf_counter()
        encoder = ChunkedRopeEncoder(
            model=config.model,
            dtype=config.kv_dtype,
            device=config.kv_device,
            max_position=config.kv_max_position,
        )
        tokenizer = encoder.tokenizer
        for user_index in range(point.num_users):
            memory = build_synthetic_memory(
                tokenizer,
                user_index=user_index,
                memory_tokens=point.memory_tokens,
                seed=config.seed,
            )
            chunk = encoder.encode_token_ids_chunk(memory.user_id, list(memory.memory_token_ids))
            kv_by_layer = encoder.compose_chunks([chunk])
            register_user_memory(
                namespace,
                user_id=memory.user_id,
                kv_by_layer=kv_by_layer,
                num_tokens=len(memory.memory_token_ids),
                token_ids=list(memory.memory_token_ids),
            )
            del chunk, kv_by_layer
            if (user_index + 1) % 10 == 0 or user_index + 1 == point.num_users:
                print(f"kv_injection: precomputed {user_index + 1}/{point.num_users} users", flush=True)
        kv_precompute_time_s = time.perf_counter() - precompute_started
        store_stats = namespace_stats(namespace)

        prompt_started = time.perf_counter()
        prompts = []
        for request in build_requests(
            num_users=point.num_users,
            requests_per_user=config.requests_per_user,
            seed=config.seed,
        ):
            memory = build_synthetic_memory(
                tokenizer,
                user_index=request.user_index,
                memory_tokens=point.memory_tokens,
                seed=config.seed,
            )
            prompts.append(
                {
                    "prompt_token_ids": build_memory_prompt(
                        tokenizer,
                        memory.memory_token_ids,
                        request.query,
                    )
                }
            )
        prompt_build_time_s = time.perf_counter() - prompt_started
        _validate_prompt_lengths(config, prompts)

        encoder.release_model()
        transfer_config = build_strict_gpu_kv_transfer_config(
            connector_module=config.kv_connector_module,
            namespace=namespace,
            default_user_id=None,
            allow_prefix_scan=True,
            log_memory_hits=False,
        )
        llm, sampling_params, engine_startup_time_s = _start_llm(
            config,
            kv_transfer_config=transfer_config,
            force_inprocess=True,
        )
        _warm_up(llm, prompts, sampling_params, config.warmup_batches)
        measured = _measure_batch(llm, prompts, sampling_params)
        return _result_row(
            config,
            point,
            condition="kv_injection",
            wall_time_s=measured["generation_time_s"],
            generation_time_s=measured["generation_time_s"],
            prompt_build_time_s=prompt_build_time_s,
            kv_precompute_time_s=kv_precompute_time_s,
            engine_startup_time_s=engine_startup_time_s,
            kv_store_gpu_mb=float(store_stats.get("total_gpu_mb", 0.0)),
            total_input_tokens=measured["total_input_tokens"],
            total_output_tokens=measured["total_output_tokens"],
        )
    finally:
        _release_llm(llm)
        if encoder is not None:
            encoder.close()
        drop_namespace(namespace)


def _run_mem0(
    config: ThroughputConfig,
    matrix: tuple[BenchmarkPoint, ...],
) -> list[dict[str, Any]]:
    if not config.embedding_api_key and not config.embedding_base_url:
        raise RuntimeError(
            "Mem0 throughput requires --embedding-api-key/OPENAI_API_KEY or a local --embedding-base-url."
        )

    from ..retrieval.mem0_provider import create_mem0_memory
    from ..retrieval.memory_builder import embed_mem0_query

    llm: Any | None = None
    open_memory: Any | None = None
    mem0_store_root = config.run_dir / "worker-results" / "mem0-stores"
    if mem0_store_root.exists():
        shutil.rmtree(mem0_store_root)
    try:
        llm, sampling_params, engine_startup_time_s = _start_llm(config)
        tokenizer = llm.get_tokenizer()
        results: list[dict[str, Any]] = []
        memory_sizes = list(dict.fromkeys(point.memory_tokens for point in matrix))
        for memory_tokens in memory_sizes:
            points = [point for point in matrix if point.memory_tokens == memory_tokens]
            max_users = max(point.num_users for point in points)
            store_root = mem0_store_root / f"tokens-{memory_tokens}"
            accumulators = {
                point: {
                    "prompts": [],
                    "retrieval_time_s": 0.0,
                    "vector_search_time_s": 0.0,
                    "prompt_build_time_s": 0.0,
                    "memory_setup_time_s": 0.0,
                }
                for point in points
            }
            requests_by_user: dict[int, list[Any]] = {user_index: [] for user_index in range(max_users)}
            for request in build_requests(
                num_users=max_users,
                requests_per_user=config.requests_per_user,
                seed=config.seed,
            ):
                requests_by_user[request.user_index].append(request)
            print(f"mem0: preparing {max_users} user stores for memory_tokens={memory_tokens}", flush=True)
            for user_index in range(max_users):
                memory_spec = build_synthetic_memory(
                    tokenizer,
                    user_index=user_index,
                    memory_tokens=memory_tokens,
                    seed=config.seed,
                )
                setup_started = time.perf_counter()
                open_memory = create_mem0_memory(
                    store_root=store_root / memory_spec.user_id,
                    vector_config=_vector_config(config),
                    embedding_model=config.embedding_model,
                    embedding_api_key=config.embedding_api_key or "not-needed",
                    embedding_base_url=config.embedding_base_url,
                )
                try:
                    for entry_index, entry in enumerate(memory_spec.entries):
                        open_memory.add(
                            [{"role": "user", "content": entry}],
                            user_id=memory_spec.user_id,
                            infer=False,
                            metadata={
                                "user_id": memory_spec.user_id,
                                "memory_tokens": memory_tokens,
                                "entry_index": entry_index,
                            },
                        )
                    _finalize_mem0(open_memory)
                except BaseException:
                    _close_mem0(open_memory)
                    open_memory = None
                    raise
                setup_time_s = time.perf_counter() - setup_started

                for point in points:
                    if user_index >= point.num_users:
                        continue
                    accumulator = accumulators[point]
                    accumulator["memory_setup_time_s"] += setup_time_s
                    for request in requests_by_user[user_index]:
                        retrieval_started = time.perf_counter()
                        query_embedding = embed_mem0_query(open_memory, request.query)
                        vector_store = getattr(open_memory, "vector_store", None)
                        search = getattr(vector_store, "search", None)
                        if not callable(search):
                            raise RuntimeError("Mem0 memory has no searchable vector_store.")
                        hits = list(
                            reversed(
                                search(
                                    query=request.query,
                                    vectors=query_embedding,
                                    top_k=config.top_k,
                                )
                            )
                        )
                        accumulator["retrieval_time_s"] += time.perf_counter() - retrieval_started
                        metrics = getattr(vector_store, "last_search_metrics", None)
                        accumulator["vector_search_time_s"] += (
                            float(getattr(metrics, "search_time_ms", 0.0) or 0.0) / 1000
                        )

                        prompt_started = time.perf_counter()
                        accumulator["prompts"].append(
                            {
                                "prompt_token_ids": build_retrieval_prompt(
                                    tokenizer,
                                    (_memory_text(hit) for hit in hits),
                                    request.query,
                                )
                            }
                        )
                        accumulator["prompt_build_time_s"] += time.perf_counter() - prompt_started

                _close_mem0(open_memory)
                open_memory = None
                if (user_index + 1) % 10 == 0 or user_index + 1 == max_users:
                    print(f"mem0: prepared {user_index + 1}/{max_users} users", flush=True)

            for point in points:
                print(f"mem0: users={point.num_users} memory_tokens={point.memory_tokens}", flush=True)
                accumulator = accumulators[point]
                prompts = accumulator["prompts"]
                _validate_prompt_lengths(config, prompts)
                _warm_up(llm, prompts, sampling_params, config.warmup_batches)
                measured = _measure_batch(llm, prompts, sampling_params)
                wall_time_s = (
                    accumulator["retrieval_time_s"]
                    + accumulator["prompt_build_time_s"]
                    + measured["generation_time_s"]
                )
                results.append(
                    _result_row(
                        config,
                        point,
                        condition="mem0",
                        vector_backend=config.vector_backend,
                        jasper_effective_beam_width=(
                            max(config.jasper_beam_width, config.top_k)
                            if config.vector_backend == "jasper"
                            else None
                        ),
                        wall_time_s=wall_time_s,
                        generation_time_s=measured["generation_time_s"],
                        retrieval_time_s=accumulator["retrieval_time_s"],
                        vector_search_time_s=accumulator["vector_search_time_s"],
                        prompt_build_time_s=accumulator["prompt_build_time_s"],
                        memory_setup_time_s=accumulator["memory_setup_time_s"],
                        engine_startup_time_s=engine_startup_time_s,
                        total_input_tokens=measured["total_input_tokens"],
                        total_output_tokens=measured["total_output_tokens"],
                    )
                )

        return results
    finally:
        if open_memory is not None:
            _close_mem0(open_memory)
        shutil.rmtree(mem0_store_root, ignore_errors=True)
        _release_llm(llm)


def _start_llm(
    config: ThroughputConfig,
    *,
    kv_transfer_config: dict[str, Any] | None = None,
    force_inprocess: bool = False,
) -> tuple[Any, Any, float]:
    from ..kv.vllm_runtime import (
        common_vllm_kwargs,
        force_vllm_inprocess_mode,
        sanitize_repo_vllm_env_for_import,
    )

    if force_inprocess:
        force_vllm_inprocess_mode()
    else:
        sanitize_repo_vllm_env_for_import()
    from vllm import LLM, SamplingParams

    kwargs = common_vllm_kwargs(config)
    kwargs["block_size"] = config.kv_block_size
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
    return llm, sampling_params, startup_time_s


def _warm_up(llm: Any, prompts: list[dict[str, list[int]]], sampling_params: Any, batches: int) -> None:
    warmup_prompts = prompts[: min(10, len(prompts))]
    for _ in range(batches):
        llm.generate(warmup_prompts, sampling_params, use_tqdm=False)


def _measure_batch(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling_params: Any,
) -> dict[str, int | float]:
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    generation_time_s = time.perf_counter() - started
    total_input_tokens = sum(len(output.prompt_token_ids) for output in outputs)
    total_output_tokens = sum(
        len(output.outputs[0].token_ids)
        for output in outputs
        if getattr(output, "outputs", None)
    )
    if len(outputs) != len(prompts):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts.")
    return {
        "generation_time_s": generation_time_s,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


def _result_row(
    config: ThroughputConfig,
    point: BenchmarkPoint,
    *,
    condition: str,
    vector_backend: str | None = None,
    jasper_effective_beam_width: int | None = None,
    wall_time_s: float,
    generation_time_s: float,
    retrieval_time_s: float = 0.0,
    vector_search_time_s: float = 0.0,
    prompt_build_time_s: float = 0.0,
    memory_setup_time_s: float = 0.0,
    kv_precompute_time_s: float = 0.0,
    engine_startup_time_s: float = 0.0,
    kv_store_gpu_mb: float = 0.0,
    total_input_tokens: int,
    total_output_tokens: int,
) -> dict[str, Any]:
    total_requests = point.num_users * config.requests_per_user
    if wall_time_s <= 0 or generation_time_s <= 0:
        raise RuntimeError("Measured benchmark time must be greater than zero.")
    row = {
        "run_id": config.run_id,
        "model": config.model,
        "model_label": config.model_label,
        "condition": condition,
        "vector_backend": vector_backend,
        "jasper_effective_beam_width": jasper_effective_beam_width,
        "num_users": point.num_users,
        "memory_tokens": point.memory_tokens,
        "requests_per_user": config.requests_per_user,
        "total_requests": total_requests,
        "wall_time_s": wall_time_s,
        "throughput_qps": total_requests / wall_time_s,
        "avg_latency_ms": wall_time_s / total_requests * 1000,
        "generation_time_s": generation_time_s,
        "retrieval_time_s": retrieval_time_s,
        "vector_search_time_s": vector_search_time_s,
        "prompt_build_time_s": prompt_build_time_s,
        "memory_setup_time_s": memory_setup_time_s,
        "kv_precompute_time_s": kv_precompute_time_s,
        "engine_startup_time_s": engine_startup_time_s,
        "kv_store_gpu_mb": kv_store_gpu_mb,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_tokens_per_second": total_input_tokens / generation_time_s,
        "output_tokens_per_second": total_output_tokens / generation_time_s,
    }
    if tuple(row) != RESULT_COLUMNS:
        raise AssertionError("Throughput result columns do not match the report schema.")
    return row


def _validate_prompt_lengths(
    config: ThroughputConfig,
    prompts: Iterable[dict[str, list[int]]],
) -> None:
    longest = max((len(prompt["prompt_token_ids"]) for prompt in prompts), default=0)
    required = longest + config.max_output_tokens
    if required > config.kv_max_model_len:
        raise RuntimeError(
            f"Longest prompt plus output is {required} tokens, exceeding max_model_len={config.kv_max_model_len}."
        )


def _vector_config(config: ThroughputConfig) -> VectorStoreConfig:
    beam_width = config.jasper_beam_width
    if config.vector_backend == "jasper":
        beam_width = max(beam_width, config.top_k)
    return VectorStoreConfig(
        backend=config.vector_backend,
        distance=config.vector_distance,
        n_neighbors=config.jasper_n_neighbors,
        alpha=config.jasper_alpha,
        workspace_budget=config.jasper_workspace_budget,
        beam_width=beam_width,
    )


def _finalize_mem0(memory: Any) -> None:
    vector_store = getattr(memory, "vector_store", None)
    finalize = getattr(vector_store, "finalize", None)
    if callable(finalize):
        finalize()


def _close_mem0(memory: Any) -> None:
    vector_store = getattr(memory, "vector_store", None)
    close = getattr(vector_store, "close", None)
    if callable(close):
        close()


def _memory_text(hit: Any) -> str:
    payload = getattr(hit, "payload", None)
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("memory") or payload.get("data") or payload.get("text") or "")


def _release_llm(llm: Any | None) -> None:
    if llm is None:
        return
    from ..kv.vllm_runtime import empty_cuda_cache

    del llm
    empty_cuda_cache(collect_ipc=True)


if __name__ == "__main__":
    main()
