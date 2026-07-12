from __future__ import annotations

import argparse
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from ..data import ConversationSample, QuestionAnswer, load_locomo
from ..kv.prompting import (
    build_kv_equivalence_prompt_token_ids,
    build_memory_prompt_token_ids,
)
from ..kv.tokenization import encode_text_no_special
from ..kv.request_identity import MEMORY_USER_ID_EXTRA_ARG
from ..results import write_json
from ..retrieval.mem0_provider import create_mem0_memory
from ..retrieval.memory_builder import embed_mem0_query, load_turns_into_memory
from ..embedding.cache import CachedEmbedder
from ..vector_types import VectorStoreConfig
from .config import (
    ALL_CONDITIONS,
    ThroughputConfig,
    condition_vector_backend,
    parse_user_counts,
)
from .reporting import RESULT_COLUMNS
from .workload import (
    LocomoRequest,
    build_locomo_requests,
    build_no_memory_prompt,
    user_id,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Internal throughput benchmark worker.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=ALL_CONDITIONS, required=True)
    parser.add_argument("--user-counts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    config = ThroughputConfig.from_json_file(args.config)
    user_counts = parse_user_counts(args.user_counts)
    unknown = [count for count in user_counts if count not in config.user_counts]
    if unknown:
        parser.error(
            "Worker user counts are not in the configured list: " + ", ".join(map(str, unknown))
        )

    results = run_condition(config, args.condition, user_counts)
    write_json(args.output, {"condition": args.condition, "results": results})
    print(f"worker wrote {len(results)} row(s) to {args.output}", flush=True)


def run_condition(
    config: ThroughputConfig,
    condition: str,
    user_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    if condition == "no_memory":
        return _run_no_memory(config, user_counts)
    if condition == "kv_injection":
        if len(user_counts) != 1:
            raise ValueError("A KV worker must receive exactly one user count.")
        return [_run_kv_injection(config, user_counts[0])]
    if condition in {"mem0_qdrant", "mem0_jasper"}:
        backend = condition_vector_backend(condition)
        if backend is None:
            raise ValueError(f"Condition {condition} has no vector backend.")
        return _run_mem0(config, user_counts, condition=condition, backend=backend)
    raise ValueError(f"Unsupported condition: {condition}")


def _run_no_memory(
    config: ThroughputConfig,
    user_counts: tuple[int, ...],
) -> list[dict[str, Any]]:
    samples = _load_samples(config)
    llm: Any | None = None
    try:
        llm, sampling_params, engine_startup_time_s = _start_llm(config)
        tokenizer = llm.get_tokenizer()
        results: list[dict[str, Any]] = []
        for num_users in user_counts:
            print(f"no_memory: users={num_users}", flush=True)
            prompt_started = time.perf_counter()
            prompts = [
                {"prompt_token_ids": build_no_memory_prompt(tokenizer, request.query)}
                for request in build_locomo_requests(
                    samples,
                    num_users=num_users,
                    requests_per_user=config.requests_per_user,
                    seed=config.seed,
                )
            ]
            prompt_build_time_s = time.perf_counter() - prompt_started
            _validate_prompt_lengths(config, prompts)
            _warm_up(llm, prompts, sampling_params, config.warmup_batches)
            measured = _measure_batch(llm, prompts, sampling_params)
            results.append(
                _result_row(
                    config,
                    num_users,
                    condition="no_memory",
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


def _run_kv_injection(config: ThroughputConfig, num_users: int) -> dict[str, Any]:
    """Jasper retrieval per request, injecting the retrieved turns' pre-encoded KV.

    This is the head-to-head counterpart of mem0_jasper: identical per-user
    stores and identical top-k searches on the critical path, but retrieved
    memories reach the model as composed KV tensors instead of prompt text.
    """
    if config.kv_device not in {"cuda", "cuda:0"}:
        raise RuntimeError("The current strict GPU registry requires --device cuda:0.")
    if not config.embedding_api_key and not config.embedding_base_url:
        raise RuntimeError(
            "kv_injection retrieval requires --embedding-api-key/OPENAI_API_KEY or a local "
            "--embedding-base-url."
        )

    from ..kv.chunked_rope import ChunkedRopeEncoder, ChunkedRopeSampleComposer
    from ..kv.gpu_registry import drop_namespace, namespace_stats, register_user_memory
    from ..kv.vllm_runtime import build_strict_gpu_kv_transfer_config, force_vllm_inprocess_mode

    force_vllm_inprocess_mode()
    samples = _load_samples(config)
    requests = build_locomo_requests(
        samples,
        num_users=num_users,
        requests_per_user=config.requests_per_user,
        seed=config.seed,
    )
    samples_by_user = {
        user_index: samples[user_index % len(samples)] for user_index in range(num_users)
    }
    used_samples = list({sample.sample_id: sample for sample in samples_by_user.values()}.values())

    namespace = f"throughput-{config.run_id}-{uuid.uuid4().hex}"
    mem0_store_root = config.run_dir / "worker-results" / f"kv-stores-{num_users}u"
    if mem0_store_root.exists():
        shutil.rmtree(mem0_store_root)
    encoder: Any | None = None
    composers: dict[str, Any] = {}
    llm: Any | None = None
    open_memory: Any | None = None
    try:
        print(f"kv_injection: users={num_users} precompute starting", flush=True)
        precompute_started = time.perf_counter()
        encoder = ChunkedRopeEncoder(
            model=config.model,
            dtype=config.kv_dtype,
            device=config.kv_device,
            max_position=config.kv_max_position,
        )
        for sample_number, sample in enumerate(used_samples, start=1):
            composer = ChunkedRopeSampleComposer(
                encoder=encoder,
                context_window=config.context_window,
            )
            composer.encode_sample(sample)
            composers[sample.sample_id] = composer
            if sample_number == 1:
                _check_kv_gpu_projection(
                    config,
                    num_users=num_users,
                    total_requests=len(requests),
                    unique_sample_count=len(used_samples),
                    first_composer=composer,
                )
            print(
                f"kv_injection: encoded sample {sample_number}/{len(used_samples)} "
                f"({len(composer.chunks)} turns, context_window={config.context_window})",
                flush=True,
            )
        encoder_probe_token_ids = encode_text_no_special(
            encoder.tokenizer, _TOKENIZER_PARITY_PROBE_TEXT
        )
        encoder.release_model()
        kv_precompute_time_s = time.perf_counter() - precompute_started

        transfer_config = build_strict_gpu_kv_transfer_config(
            connector_module=config.kv_connector_module,
            namespace=namespace,
            default_user_id=None,
            allow_prefix_scan=False,
        )
        llm, sampling_params, engine_startup_time_s = _start_llm(
            config,
            kv_transfer_config=transfer_config,
        )
        tokenizer = llm.get_tokenizer()
        _require_tokenizer_parity(encoder_probe_token_ids, tokenizer)

        memory_setup_time_s = 0.0
        retrieval_time_s = 0.0
        vector_search_time_s = 0.0
        kv_compose_time_s = 0.0
        kv_verify_time_s = 0.0
        prompt_build_time_s = 0.0
        memory_turn_total = 0
        prompts: list[dict[str, list[int]]] = []
        prompt_memory_user_ids: list[str] = []
        request_index = 0
        requests_by_user: dict[int, list[LocomoRequest]] = {
            user_index: [] for user_index in range(num_users)
        }
        for request in requests:
            requests_by_user[request.user_index].append(request)

        for user_index in range(num_users):
            sample = samples_by_user[user_index]
            composer = composers[sample.sample_id]
            memory_turn_total += len(sample.turns)
            setup_started = time.perf_counter()
            open_memory = _build_user_store(
                config,
                backend="jasper",
                store_root=mem0_store_root / user_id(user_index),
                sample=sample,
            )
            memory_setup_time_s += time.perf_counter() - setup_started
            try:
                for request in requests_by_user[user_index]:
                    hits, elapsed_s, search_s = _search_store(
                        open_memory,
                        request.query,
                        top_k=config.top_k,
                    )
                    retrieval_time_s += elapsed_s
                    vector_search_time_s += search_s

                    compose_started = time.perf_counter()
                    composed = composer.compose(hits)
                    memory_user_id = f"request-{request_index:05d}"
                    register_user_memory(
                        namespace,
                        user_id=memory_user_id,
                        kv_by_layer=composed.kv_by_layer,
                        num_tokens=composed.num_tokens,
                        token_ids=composed.token_ids,
                    )
                    kv_compose_time_s += time.perf_counter() - compose_started

                    # Token-equivalence verification is benchmark bookkeeping,
                    # not part of the serving path; its cost is reported
                    # separately and excluded from wall_time_s.
                    verify_started = time.perf_counter()
                    canonical_memory = build_memory_prompt_token_ids(tokenizer, sample, hits)
                    _require_canonical_memory_tokens(composed.token_ids, canonical_memory.token_ids)
                    kv_verify_time_s += time.perf_counter() - verify_started

                    prompt_started = time.perf_counter()
                    qa = _request_question_answer(request)
                    prompts.append(
                        {
                            "prompt_token_ids": build_kv_equivalence_prompt_token_ids(
                                tokenizer,
                                composed.token_ids,
                                sample,
                                qa,
                            ).prompt_token_ids
                        }
                    )
                    prompt_memory_user_ids.append(memory_user_id)
                    prompt_build_time_s += time.perf_counter() - prompt_started
                    request_index += 1
            finally:
                _close_mem0(open_memory)
                open_memory = None
            if (user_index + 1) % 10 == 0 or user_index + 1 == num_users:
                print(f"kv_injection: retrieved {user_index + 1}/{num_users} users", flush=True)

        store_stats = namespace_stats(namespace)
        _validate_prompt_lengths(config, prompts)
        routed_sampling_params = _sampling_params_with_memory_user_ids(
            sampling_params,
            prompt_memory_user_ids,
        )
        _warm_up(llm, prompts, routed_sampling_params, config.warmup_batches)
        measured = _measure_batch(llm, prompts, routed_sampling_params)
        wall_time_s = (
            retrieval_time_s
            + kv_compose_time_s
            + prompt_build_time_s
            + measured["generation_time_s"]
        )
        return _result_row(
            config,
            num_users,
            condition="kv_injection",
            vector_backend="jasper",
            jasper_effective_beam_width=max(config.jasper_beam_width, config.top_k),
            memory_turn_count=memory_turn_total / num_users,
            wall_time_s=wall_time_s,
            generation_time_s=measured["generation_time_s"],
            retrieval_time_s=retrieval_time_s,
            vector_search_time_s=vector_search_time_s,
            prompt_build_time_s=prompt_build_time_s,
            memory_setup_time_s=memory_setup_time_s,
            kv_precompute_time_s=kv_precompute_time_s,
            kv_compose_time_s=kv_compose_time_s,
            kv_verify_time_s=kv_verify_time_s,
            engine_startup_time_s=engine_startup_time_s,
            kv_store_gpu_mb=float(store_stats.get("total_gpu_mb", 0.0)),
            total_input_tokens=measured["total_input_tokens"],
            total_output_tokens=measured["total_output_tokens"],
        )
    finally:
        if open_memory is not None:
            _close_mem0(open_memory)
        _release_llm(llm)
        for composer in composers.values():
            composer.close()
        if encoder is not None:
            encoder.close()
        drop_namespace(namespace)
        shutil.rmtree(mem0_store_root, ignore_errors=True)


def _run_mem0(
    config: ThroughputConfig,
    user_counts: tuple[int, ...],
    *,
    condition: str,
    backend: str,
) -> list[dict[str, Any]]:
    if not config.embedding_api_key and not config.embedding_base_url:
        raise RuntimeError(
            f"{condition} requires --embedding-api-key/OPENAI_API_KEY or a local --embedding-base-url."
        )

    samples = _load_samples(config)

    llm: Any | None = None
    open_memory: Any | None = None
    mem0_store_root = config.run_dir / "worker-results" / f"{condition}-stores"
    if mem0_store_root.exists():
        shutil.rmtree(mem0_store_root)
    try:
        llm, sampling_params, engine_startup_time_s = _start_llm(config)
        tokenizer = llm.get_tokenizer()
        results: list[dict[str, Any]] = []
        max_users = max(user_counts)
        accumulators = {
            count: {
                "prompts": [],
                "retrieval_time_s": 0.0,
                "vector_search_time_s": 0.0,
                "prompt_build_time_s": 0.0,
                "memory_setup_time_s": 0.0,
                "memory_turn_total": 0,
            }
            for count in user_counts
        }
        requests_by_user: dict[int, list[LocomoRequest]] = {
            user_index: [] for user_index in range(max_users)
        }
        for request in build_locomo_requests(
            samples,
            num_users=max_users,
            requests_per_user=config.requests_per_user,
            seed=config.seed,
        ):
            requests_by_user[request.user_index].append(request)

        print(f"{condition}: preparing {max_users} user stores", flush=True)
        for user_index in range(max_users):
            sample = samples[user_index % len(samples)]

            setup_started = time.perf_counter()
            open_memory = _build_user_store(
                config,
                backend=backend,
                store_root=mem0_store_root / user_id(user_index),
                sample=sample,
            )
            setup_time_s = time.perf_counter() - setup_started

            try:
                for count in user_counts:
                    if user_index >= count:
                        continue
                    accumulator = accumulators[count]
                    accumulator["memory_setup_time_s"] += setup_time_s
                    accumulator["memory_turn_total"] += len(sample.turns)
                    for request in requests_by_user[user_index]:
                        hits, elapsed_s, search_s = _search_store(
                            open_memory,
                            request.query,
                            top_k=config.top_k,
                        )
                        accumulator["retrieval_time_s"] += elapsed_s
                        accumulator["vector_search_time_s"] += search_s

                        prompt_started = time.perf_counter()
                        memory_prompt = build_memory_prompt_token_ids(tokenizer, sample, hits)
                        qa = _request_question_answer(request)
                        accumulator["prompts"].append(
                            {
                                "prompt_token_ids": build_kv_equivalence_prompt_token_ids(
                                    tokenizer,
                                    memory_prompt.token_ids,
                                    sample,
                                    qa,
                                ).prompt_token_ids
                            }
                        )
                        accumulator["prompt_build_time_s"] += time.perf_counter() - prompt_started
            finally:
                _close_mem0(open_memory)
                open_memory = None
            if (user_index + 1) % 10 == 0 or user_index + 1 == max_users:
                print(f"{condition}: prepared {user_index + 1}/{max_users} users", flush=True)

        for count in user_counts:
            print(f"{condition}: users={count}", flush=True)
            accumulator = accumulators[count]
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
                    count,
                    condition=condition,
                    vector_backend=backend,
                    jasper_effective_beam_width=(
                        max(config.jasper_beam_width, config.top_k)
                        if backend == "jasper"
                        else None
                    ),
                    memory_turn_count=accumulator["memory_turn_total"] / count,
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


def _load_samples(config: ThroughputConfig) -> list[ConversationSample]:
    samples = load_locomo(config.dataset_path)
    if not samples:
        raise RuntimeError(f"No LoCoMo samples found in {config.dataset_path}.")
    return samples


def _build_user_store(
    config: ThroughputConfig,
    *,
    backend: str,
    store_root: Path,
    sample: ConversationSample,
) -> Any:
    """Ingest a sample's raw turns into a fresh per-user store.

    Turn embeddings go through the shared cache (free and offline after
    --preembed-only), but the raw embedder is restored before returning so
    query-time retrieval measures live embedding latency, not cache reads.
    """
    memory = create_mem0_memory(
        store_root=store_root,
        vector_config=_vector_config(config, backend),
        embedding_model=config.embedding_model,
        embedding_api_key=config.embedding_api_key or "not-needed",
        embedding_base_url=config.embedding_base_url,
    )
    raw_embedder = getattr(memory, "embedding_model", None) or getattr(memory, "embedder", None)
    if config.embedding_cache_enabled and raw_embedder is not None:
        cached = CachedEmbedder(
            raw_embedder,
            cache_dir=config.embedding_cache_dir,
            model=config.embedding_model,
            mode="write",
            endpoint=config.embedding_base_url,
        )
        if hasattr(memory, "embedding_model"):
            memory.embedding_model = cached
        if hasattr(memory, "embedder"):
            memory.embedder = cached
    try:
        load_turns_into_memory(memory, sample)
        _finalize_mem0(memory)
    except BaseException:
        _close_mem0(memory)
        raise
    if raw_embedder is not None:
        if hasattr(memory, "embedding_model"):
            memory.embedding_model = raw_embedder
        if hasattr(memory, "embedder"):
            memory.embedder = raw_embedder
    return memory


def _search_store(memory: Any, query: str, *, top_k: int) -> tuple[list[Any], float, float]:
    """Embed the query and search the store; returns (hits, elapsed_s, backend_search_s).

    Hits are reversed to weakest-first, mirroring the accuracy bench, so the
    shared prompt builder and KV composer place the strongest memory closest
    to the question.
    """
    retrieval_started = time.perf_counter()
    query_embedding = embed_mem0_query(memory, query)
    vector_store = getattr(memory, "vector_store", None)
    search = getattr(vector_store, "search", None)
    if not callable(search):
        raise RuntimeError("Mem0 memory has no searchable vector_store.")
    hits = list(
        reversed(
            search(
                query=query,
                vectors=query_embedding,
                top_k=top_k,
            )
        )
    )
    elapsed_s = time.perf_counter() - retrieval_started
    metrics = getattr(vector_store, "last_search_metrics", None)
    search_s = float(getattr(metrics, "search_time_ms", 0.0) or 0.0) / 1000
    return hits, elapsed_s, search_s


def _request_question_answer(request: LocomoRequest) -> QuestionAnswer:
    return QuestionAnswer(
        sample_id=request.sample_id,
        question_id=request.question_id,
        question=request.query,
        answer="",
        category="",
    )


# Mixed-script probe so tokenizer stacks that differ in template, whitespace,
# byte-fallback, or unicode handling cannot encode it identically by accident.
_TOKENIZER_PARITY_PROBE_TEXT = (
    "SPEAKER Caroline (2023-05-08): I'll re-check trip #42 — cost $1,300.50; "
    "email caroline@example.com, emoji 🙂, CJK 你好, newline\nend.\n"
)


def _require_tokenizer_parity(
    encoder_probe_token_ids: list[int],
    engine_tokenizer: Any,
) -> None:
    """The connector matches chunk ids as a prompt prefix, so the encoder and
    engine tokenizers must produce identical ids."""
    engine_probe_token_ids = encode_text_no_special(
        engine_tokenizer, _TOKENIZER_PARITY_PROBE_TEXT
    )
    if encoder_probe_token_ids == engine_probe_token_ids:
        return
    raise RuntimeError(
        "Encoder and engine tokenizers disagree; KV chunk token ids will not "
        "match prompt token ids and injection cannot work. "
        f"encoder_probe_tokens={len(encoder_probe_token_ids)} "
        f"engine_probe_tokens={len(engine_probe_token_ids)} "
        f"engine_tokenizer={type(engine_tokenizer).__name__}. "
        "Ensure ChunkedRopeEncoder loads its tokenizer via "
        "vllm.transformers_utils.tokenizer.get_tokenizer and that the "
        "transformers/vllm/mistral_common versions match the engine's."
    )


def _require_canonical_memory_tokens(
    composed_token_ids: list[int],
    canonical_token_ids: list[int],
) -> None:
    """Composed KV memory tokens must equal the shared prompt-injection layout."""
    if composed_token_ids == canonical_token_ids:
        return
    mismatch_index = next(
        (
            index
            for index, (left, right) in enumerate(
                zip(composed_token_ids, canonical_token_ids)
            )
            if left != right
        ),
        min(len(composed_token_ids), len(canonical_token_ids)),
    )
    raise RuntimeError(
        "Composed KV memory tokens differ from the canonical memory section at "
        f"index={mismatch_index}: composed_length={len(composed_token_ids)} "
        f"canonical_length={len(canonical_token_ids)}."
    )


def _check_kv_gpu_projection(
    config: ThroughputConfig,
    *,
    num_users: int,
    total_requests: int,
    unique_sample_count: int,
    first_composer: Any,
) -> None:
    """Fail before the expensive precompute if the KV footprint cannot fit.

    Sample chunks stay resident for the whole run (replicas share them), so
    peak GPU usage is the vLLM pool plus all sample chunks plus all
    per-request composed copies.
    """
    import torch

    from ..kv.gpu_memory_store import _tensor_nbytes

    def chunk_bytes(chunk: Any) -> int:
        return sum(_tensor_nbytes(tensor) for tensor in chunk.kv_by_layer.values())

    turn_chunks = dict(first_composer.chunks)
    first_sample_bytes = sum(chunk_bytes(chunk) for chunk in turn_chunks.values())
    first_sample_tokens = sum(len(chunk.token_ids) for chunk in turn_chunks.values())
    if first_sample_tokens <= 0 or not turn_chunks:
        return
    bytes_per_token = first_sample_bytes / first_sample_tokens

    scaffold_tokens = len(first_composer.header_chunk.token_ids)
    if first_composer.footer_chunk is not None:
        scaffold_tokens += len(first_composer.footer_chunk.token_ids)
    retrieved_turns = min(config.top_k, len(turn_chunks))
    average_turn_tokens = first_sample_tokens / len(turn_chunks)
    composed_tokens = scaffold_tokens + retrieved_turns * average_turn_tokens
    composed_bytes = total_requests * composed_tokens * bytes_per_token
    source_bytes = first_sample_bytes * unique_sample_count

    device_total = torch.cuda.get_device_properties(0).total_memory
    vllm_pool = config.kv_gpu_memory_utilization * device_total
    projected_peak = vllm_pool + composed_bytes + source_bytes
    budget = 0.97 * device_total
    if projected_peak > budget:
        raise RuntimeError(
            "Projected KV GPU footprint exceeds device memory: "
            f"vllm_pool={vllm_pool / 2**30:.1f}GiB "
            f"composed={composed_bytes / 2**30:.1f}GiB "
            f"sources={source_bytes / 2**30:.1f}GiB "
            f"device={device_total / 2**30:.1f}GiB "
            f"(users={num_users}, requests={total_requests}). "
            "Shrink the user count, lower --top-k, or reduce "
            "--gpu-memory-utilization."
        )


def _start_llm(
    config: ThroughputConfig,
    *,
    kv_transfer_config: dict[str, Any] | None = None,
) -> tuple[Any, Any, float]:
    from ..kv.vllm_runtime import common_vllm_kwargs, force_vllm_inprocess_mode

    # Every condition measures the same in-process V1 engine; otherwise QPS
    # differences would be confounded by engine architecture.
    force_vllm_inprocess_mode()
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


def _sampling_params_with_memory_user_ids(
    sampling_params: Any,
    memory_user_ids: list[str],
) -> list[Any]:
    routed: list[Any] = []
    for memory_user_id in memory_user_ids:
        clone = getattr(sampling_params, "clone", None)
        if not callable(clone):
            raise RuntimeError("Pinned vLLM SamplingParams must provide clone() for request routing.")
        request_params = clone()
        extra_args = dict(getattr(request_params, "extra_args", None) or {})
        extra_args[MEMORY_USER_ID_EXTRA_ARG] = memory_user_id
        request_params.extra_args = extra_args
        routed.append(request_params)
    return routed


def _warm_up(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling_params: Any | list[Any],
    batches: int,
) -> None:
    warmup_prompts = prompts[: min(10, len(prompts))]
    warmup_sampling_params = (
        sampling_params[: len(warmup_prompts)]
        if isinstance(sampling_params, list)
        else sampling_params
    )
    for _ in range(batches):
        llm.generate(warmup_prompts, warmup_sampling_params, use_tqdm=False)


def _measure_batch(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling_params: Any | list[Any],
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
    num_users: int,
    *,
    condition: str,
    vector_backend: str | None = None,
    jasper_effective_beam_width: int | None = None,
    memory_turn_count: float = 0.0,
    wall_time_s: float,
    generation_time_s: float,
    retrieval_time_s: float = 0.0,
    vector_search_time_s: float = 0.0,
    prompt_build_time_s: float = 0.0,
    memory_setup_time_s: float = 0.0,
    kv_precompute_time_s: float = 0.0,
    kv_compose_time_s: float = 0.0,
    kv_verify_time_s: float = 0.0,
    engine_startup_time_s: float = 0.0,
    kv_store_gpu_mb: float = 0.0,
    total_input_tokens: int,
    total_output_tokens: int,
) -> dict[str, Any]:
    total_requests = num_users * config.requests_per_user
    if wall_time_s <= 0 or generation_time_s <= 0:
        raise RuntimeError("Measured benchmark time must be greater than zero.")
    row = {
        "run_id": config.run_id,
        "model": config.model,
        "model_label": config.model_label,
        "condition": condition,
        "vector_backend": vector_backend,
        "jasper_effective_beam_width": jasper_effective_beam_width,
        "num_users": num_users,
        "memory_turn_count": memory_turn_count,
        "requests_per_user": config.requests_per_user,
        "total_requests": total_requests,
        "wall_time_s": wall_time_s,
        "throughput_qps": total_requests / wall_time_s,
        "avg_latency_ms": wall_time_s / total_requests * 1000,
        "generation_time_s": generation_time_s,
        "retrieval_time_s": retrieval_time_s,
        "vector_search_time_s": vector_search_time_s,
        "prompt_build_time_s": prompt_build_time_s,
        "kv_compose_time_s": kv_compose_time_s,
        "kv_verify_time_s": kv_verify_time_s,
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


def _vector_config(config: ThroughputConfig, backend: str) -> VectorStoreConfig:
    beam_width = config.jasper_beam_width
    if backend == "jasper":
        beam_width = max(beam_width, config.top_k)
    return VectorStoreConfig(
        backend=backend,
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


def _release_llm(llm: Any | None) -> None:
    if llm is None:
        return
    from ..kv.vllm_runtime import empty_cuda_cache

    del llm
    empty_cuda_cache(collect_ipc=True)


if __name__ == "__main__":
    main()
