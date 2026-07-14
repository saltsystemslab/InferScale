from __future__ import annotations

import argparse
import os
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

from ..data import ConversationSample, QuestionAnswer, load_locomo
from ..kv.context import (
    build_fact_context_encoding_plan,
    reverse_ranked_memory_facts,
    unique_memory_facts,
)
from ..kv.prompting import (
    build_kv_equivalence_prompt_token_ids,
    build_memory_prompt_token_ids,
    extract_memory_scaffold_token_ids,
    format_memory_fact,
)
from ..kv.connector_utils import MEMORY_USER_ID_EXTRA_ARG
from ..kv.tokenization import encode_text_no_special
from ..results import write_json
from ..retrieval.fact_catalog import FactCatalogStore, MemoryFact, fact_catalog_hits
from ..retrieval.mem0_provider import MEMORY_LLM_TEMPERATURE, create_mem0_memory
from ..retrieval.memory_builder import embed_mem0_query, load_facts_into_memory
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
            _warm_up(llm, prompts, sampling_params, config.warmup_batches, seed=config.seed)
            measured = _measure_batch(llm, prompts, sampling_params)
            results.append(
                _result_row(
                    config,
                    num_users,
                    condition="no_memory",
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
    """Jasper retrieval per request, injecting the retrieved facts' pre-encoded KV.

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

    import importlib

    import torch

    from ..kv.chunked_rope import ChunkedRopeEncoder
    from ..kv.gpu_registry import (
        drop_namespace,
        get_gpu_memory_store,
        namespace_bench_summary,
        namespace_stats,
        register_user_memory,
        reset_namespace_bench_metrics,
    )
    from ..kv.vllm_runtime import build_strict_gpu_kv_transfer_config, force_vllm_inprocess_mode

    connector_module = importlib.import_module(config.kv_connector_module)
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
    catalog_store = _fact_catalog_store(config)
    facts_by_sample = {
        sample.sample_id: catalog_store.load(sample) for sample in used_samples
    }

    namespace = f"throughput-{config.run_id}-{uuid.uuid4().hex}"
    # Create the namespace's store with the configured backend before any
    # registration; the in-process connector resolves the same namespace.
    get_gpu_memory_store(
        namespace,
        backend=config.kv_store_backend,
        num_staging_slots=config.kv_staging_slots,
    )
    mem0_store_root = config.run_dir / "worker-results" / f"kv-stores-{num_users}u"
    if mem0_store_root.exists():
        shutil.rmtree(mem0_store_root)
    encoder: Any | None = None
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
        tokenizer = encoder.tokenizer
        scaffold = extract_memory_scaffold_token_ids(
            tokenizer,
            used_samples[0],
            block_size=config.kv_block_size,
        )
        header_chunk = encoder.encode_token_ids_chunk("scaffold:header", scaffold.header_token_ids)
        footer_chunk = encoder.encode_token_ids_chunk("scaffold:footer", scaffold.footer_token_ids)
        chunks_by_sample: dict[str, dict[str, Any]] = {}
        for sample_number, sample in enumerate(used_samples, start=1):
            kv_facts = unique_memory_facts(fact_catalog_hits(facts_by_sample[sample.sample_id]))
            fact_token_ids = {
                fact.memory_id: encode_text_no_special(tokenizer, format_memory_fact(fact))
                for fact in kv_facts
            }
            fact_chunks: dict[str, Any] = {}
            for fact in kv_facts:
                # Chunk values are conditioned on the facts extracted from the
                # preceding context_window turns; only the fact-token KV slice
                # is kept (prefix-discard), matching the accuracy path's
                # vllm-kv encoding semantics.
                plan = build_fact_context_encoding_plan(
                    fact,
                    sample,
                    context_window=config.context_window,
                    max_input_tokens=config.kv_max_position,
                    fact_token_ids=fact_token_ids,
                    sample_facts=kv_facts,
                )
                fact_chunks[fact.memory_id] = encoder.encode_fact_chunk(plan)
            chunks_by_sample[sample.sample_id] = fact_chunks
            if sample_number == 1:
                _check_kv_gpu_projection(
                    config,
                    num_users=num_users,
                    total_requests=len(requests),
                    unique_sample_count=len(used_samples),
                    scaffold_chunks=(header_chunk, footer_chunk),
                    first_sample_chunks=fact_chunks,
                )
            print(
                f"kv_injection: encoded sample {sample_number}/{len(used_samples)} "
                f"({len(fact_chunks)} facts)",
                flush=True,
            )
        encoder_probe_token_ids = encode_text_no_special(
            encoder.tokenizer, _TOKENIZER_PARITY_PROBE_TEXT
        )
        encoder.release_model()
        torch.cuda.synchronize()
        kv_precompute_time_s = time.perf_counter() - precompute_started

        transfer_config = build_strict_gpu_kv_transfer_config(
            connector_module=config.kv_connector_module,
            namespace=namespace,
            default_user_id=None,
            # Requests carry their memory's user id via SamplingParams
            # extra_args, so matching is a direct lookup; scanning every
            # registered user per request is both slow and ambiguous for
            # duplicate memory sequences.
            allow_prefix_scan=False,
            log_memory_hits=False,
            store_backend=config.kv_store_backend,
            num_staging_slots=config.kv_staging_slots,
        )
        llm, sampling_params, engine_startup_time_s = _start_llm(
            config,
            kv_transfer_config=transfer_config,
        )
        _require_tokenizer_parity(encoder_probe_token_ids, llm.get_tokenizer())

        memory_setup_time_s = 0.0
        retrieval_time_s = 0.0
        vector_search_time_s = 0.0
        kv_compose_time_s = 0.0
        kv_store_write_time_s = 0.0
        kv_verify_time_s = 0.0
        prompt_build_time_s = 0.0
        fact_count_total = 0
        prompts: list[dict[str, list[int]]] = []
        prompt_memory_user_ids: list[str] = []
        first_memory_token_ids: list[int] | None = None
        first_memory_user_id: str | None = None
        request_index = 0
        requests_by_user: dict[int, list[LocomoRequest]] = {
            user_index: [] for user_index in range(num_users)
        }
        for request in requests:
            requests_by_user[request.user_index].append(request)

        for user_index in range(num_users):
            sample = samples_by_user[user_index]
            facts = facts_by_sample[sample.sample_id]
            fact_count_total += len(facts)
            setup_started = time.perf_counter()
            open_memory = _build_user_store(
                config,
                backend="jasper",
                store_root=mem0_store_root / user_id(user_index),
                facts=facts,
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

                    # Compose launches asynchronous CUDA work; synchronize on
                    # both sides of the timer so it measures the GPU time, not
                    # just the kernel-enqueue time.
                    torch.cuda.synchronize()
                    compose_started = time.perf_counter()
                    selected_facts = reverse_ranked_memory_facts(hits)
                    selected = _select_chunks_for_fact_ids(
                        [fact.memory_id for fact in selected_facts],
                        chunks_by_sample[sample.sample_id],
                    )
                    composed_chunks = [header_chunk, *selected, footer_chunk]
                    kv_by_layer = encoder.compose_chunks(composed_chunks)
                    memory_token_ids = [
                        token_id
                        for chunk in composed_chunks
                        for token_id in chunk.token_ids
                    ]
                    torch.cuda.synchronize()
                    kv_compose_time_s += time.perf_counter() - compose_started

                    # Registration is a store write (a D2H copy into pinned
                    # RAM for the cpu-pinned backend); timed apart from
                    # compose so the PCIe write cost is visible.
                    store_write_started = time.perf_counter()
                    memory_user_id = f"request-{request_index:05d}"
                    if first_memory_token_ids is None:
                        first_memory_token_ids = list(memory_token_ids)
                        first_memory_user_id = memory_user_id
                    register_user_memory(
                        namespace,
                        user_id=memory_user_id,
                        kv_by_layer=kv_by_layer,
                        num_tokens=len(memory_token_ids),
                        token_ids=memory_token_ids,
                    )
                    torch.cuda.synchronize()
                    kv_store_write_time_s += time.perf_counter() - store_write_started

                    # Token-equivalence verification is benchmark bookkeeping,
                    # not part of the serving path; its cost is reported
                    # separately and excluded from QPS.
                    verify_started = time.perf_counter()
                    canonical_memory = build_memory_prompt_token_ids(
                        tokenizer,
                        sample,
                        hits,
                        context_window=0,
                        memory_scaffold=scaffold,
                    )
                    _require_canonical_memory_tokens(memory_token_ids, canonical_memory.token_ids)
                    kv_verify_time_s += time.perf_counter() - verify_started

                    prompt_started = time.perf_counter()
                    qa = _request_question_answer(request)
                    prompts.append(
                        {
                            "prompt_token_ids": build_kv_equivalence_prompt_token_ids(
                                tokenizer,
                                memory_token_ids,
                                sample,
                                qa,
                                memory_scaffold=scaffold,
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
        # With prefix scan off, the KV warmup prompt reaches its memory via
        # the same explicit routing as measured requests; the random warmup
        # prompts stay unrouted so they match nothing.
        kv_warmup: tuple[dict[str, list[int]], Any] | None = None
        if first_memory_token_ids and first_memory_user_id:
            kv_warmup = (
                _build_kv_warmup_prompt(
                    first_memory_token_ids,
                    vocab_size=_tokenizer_vocab_size(tokenizer),
                    seed=config.seed,
                ),
                _sampling_params_with_memory_user_ids(
                    sampling_params, [first_memory_user_id]
                )[0],
            )
        _warm_up(
            llm,
            prompts,
            sampling_params,
            config.warmup_batches,
            seed=config.seed,
            kv_warmup=kv_warmup,
        )
        connector_module.reset_load_stats()
        reset_namespace_bench_metrics(namespace)
        measured = _measure_batch(llm, prompts, routed_sampling_params)
        load_stats = connector_module.snapshot_load_stats()
        bench_summary = namespace_bench_summary(namespace)
        # The store owns how its staging pool is sized and reports its
        # steady-state HBM footprint in the bench summary; the GPU store has
        # no staging pool, so fall back to its resident total.
        kv_store_gpu_mb = float(
            bench_summary.get("steady_state_staging_mb")
            or store_stats.get("total_gpu_mb", 0.0)
        )
        # requests_covered counts distinct request ids that either had memory
        # injected or whose memory region was fully served by the native
        # prefix cache (a legitimate no-load), so the check stays exact per
        # request even with prefix caching enabled and preemption re-loads.
        if load_stats["requests_covered"] < len(prompts):
            raise RuntimeError(
                "KV memory covered "
                f"{load_stats['requests_covered']} of {len(prompts)} requests "
                "(injected or natively cached); "
                "some requests generated without their memory."
            )
        return _result_row(
            config,
            num_users,
            condition="kv_injection",
            vector_backend="jasper",
            jasper_effective_beam_width=max(config.jasper_beam_width, config.top_k),
            fact_count=fact_count_total / num_users,
            generation_time_s=measured["generation_time_s"],
            retrieval_time_s=retrieval_time_s,
            vector_search_time_s=vector_search_time_s,
            prompt_build_time_s=prompt_build_time_s,
            memory_setup_time_s=memory_setup_time_s,
            kv_precompute_time_s=kv_precompute_time_s,
            kv_compose_time_s=kv_compose_time_s,
            kv_verify_time_s=kv_verify_time_s,
            engine_startup_time_s=engine_startup_time_s,
            kv_store_gpu_mb=kv_store_gpu_mb,
            kv_store_backend=config.kv_store_backend,
            kv_store_host_mb=float(store_stats.get("total_host_mb", 0.0)),
            kv_store_write_time_s=kv_store_write_time_s,
            kv_h2d_bytes=int(bench_summary.get("total_bytes_transferred", 0)),
            kv_h2d_avg_ms=float(bench_summary.get("avg_h2d_latency_ms", 0.0)),
            kv_h2d_p95_ms=float(bench_summary.get("p95_h2d_latency_ms", 0.0)),
            kv_h2d_overlap_ratio=float(bench_summary.get("avg_overlap_ratio", 0.0)),
            kv_staging_stall_ms=float(bench_summary.get("total_staging_stall_ms", 0.0)),
            kv_requests_loaded=int(load_stats["requests_loaded"]),
            total_input_tokens=measured["total_input_tokens"],
            total_output_tokens=measured["total_output_tokens"],
        )
    finally:
        if open_memory is not None:
            _close_mem0(open_memory)
        _release_llm(llm)
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
    catalog_store = _fact_catalog_store(config)
    facts_by_sample: dict[str, tuple[MemoryFact, ...]] = {}

    llm: Any | None = None
    open_memory: Any | None = None
    mem0_store_root = config.run_dir / "worker-results" / f"{condition}-stores"
    if mem0_store_root.exists():
        shutil.rmtree(mem0_store_root)
    try:
        llm, sampling_params, engine_startup_time_s = _start_llm(config)
        tokenizer = llm.get_tokenizer()
        scaffold = extract_memory_scaffold_token_ids(
            tokenizer,
            samples[0],
            block_size=config.kv_block_size,
        )
        results: list[dict[str, Any]] = []
        max_users = max(user_counts)
        accumulators = {
            count: {
                "prompts": [],
                "retrieval_time_s": 0.0,
                "vector_search_time_s": 0.0,
                "prompt_build_time_s": 0.0,
                "memory_setup_time_s": 0.0,
                "fact_count_total": 0,
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
            if sample.sample_id not in facts_by_sample:
                facts_by_sample[sample.sample_id] = catalog_store.load(sample)
            facts = facts_by_sample[sample.sample_id]

            setup_started = time.perf_counter()
            open_memory = _build_user_store(
                config,
                backend=backend,
                store_root=mem0_store_root / user_id(user_index),
                facts=facts,
            )
            setup_time_s = time.perf_counter() - setup_started

            try:
                for count in user_counts:
                    if user_index >= count:
                        continue
                    accumulator = accumulators[count]
                    accumulator["memory_setup_time_s"] += setup_time_s
                    accumulator["fact_count_total"] += len(facts)
                    for request in requests_by_user[user_index]:
                        hits, elapsed_s, search_s = _search_store(
                            open_memory,
                            request.query,
                            top_k=config.top_k,
                        )
                        accumulator["retrieval_time_s"] += elapsed_s
                        accumulator["vector_search_time_s"] += search_s

                        prompt_started = time.perf_counter()
                        memory_prompt = build_memory_prompt_token_ids(
                            tokenizer,
                            sample,
                            hits,
                            context_window=0,
                            memory_scaffold=scaffold,
                        )
                        qa = _request_question_answer(request)
                        accumulator["prompts"].append(
                            {
                                "prompt_token_ids": build_kv_equivalence_prompt_token_ids(
                                    tokenizer,
                                    memory_prompt.token_ids,
                                    sample,
                                    qa,
                                    memory_scaffold=scaffold,
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
            _warm_up(llm, prompts, sampling_params, config.warmup_batches, seed=config.seed)
            measured = _measure_batch(llm, prompts, sampling_params)
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
                    fact_count=accumulator["fact_count_total"] / count,
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


def _fact_catalog_store(config: ThroughputConfig) -> FactCatalogStore:
    return FactCatalogStore(
        config.memory_llm_cache_dir,
        provider=config.memory_llm_provider,
        model=config.memory_llm_model,
        endpoint=config.memory_llm_base_url,
        embedding_model=config.embedding_model,
        embedding_endpoint=config.embedding_base_url,
        temperature=MEMORY_LLM_TEMPERATURE,
    )


def _build_user_store(
    config: ThroughputConfig,
    *,
    backend: str,
    store_root: Path,
    facts: tuple[MemoryFact, ...],
) -> Any:
    """Replay a sample's fact catalog into a fresh per-user store.

    Fact embeddings go through the shared cache (free and offline after
    --preembed-only), but the raw embedder is restored before returning so
    query-time retrieval measures live embedding latency, not cache reads.
    """
    memory = create_mem0_memory(
        store_root=store_root,
        vector_config=_vector_config(config, backend),
        embedding_model=config.embedding_model,
        embedding_api_key=config.embedding_api_key or "not-needed",
        embedding_base_url=config.embedding_base_url,
        memory_llm_provider=config.memory_llm_provider,
        memory_llm_model=config.memory_llm_model,
        memory_llm_base_url=config.memory_llm_base_url,
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
        load_facts_into_memory(memory, facts)
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

    Hits remain best-first; the shared accuracy prompt builder reverses them
    exactly once when it constructs the injected memory sequence.
    """
    retrieval_started = time.perf_counter()
    query_embedding = embed_mem0_query(memory, query)
    vector_store = getattr(memory, "vector_store", None)
    search = getattr(vector_store, "search", None)
    if not callable(search):
        raise RuntimeError("Mem0 memory has no searchable vector_store.")
    hits = list(
        search(
            query=query,
            vectors=query_embedding,
            top_k=top_k,
        )
    )
    elapsed_s = time.perf_counter() - retrieval_started
    metrics = getattr(vector_store, "last_search_metrics", None)
    search_s = float(getattr(metrics, "search_time_ms", 0.0) or 0.0) / 1000
    return hits, elapsed_s, search_s


def _select_chunks_for_fact_ids(
    fact_ids: list[str],
    chunks_by_fact_id: dict[str, Any],
) -> list[Any]:
    """Map reverse-ranked fact ids to their pre-encoded KV chunks."""
    if not fact_ids:
        raise RuntimeError("Retrieval returned no facts for a kv_injection request.")
    selected = []
    for fact_id in fact_ids:
        chunk = chunks_by_fact_id.get(fact_id)
        if chunk is None:
            raise RuntimeError(
                f"Retrieved fact_id={fact_id} has no pre-encoded KV chunk."
            )
        selected.append(chunk)
    return selected


def _request_question_answer(request: LocomoRequest) -> QuestionAnswer:
    return QuestionAnswer(
        sample_id=request.sample_id,
        question_id=request.question_id,
        question=request.query,
        answer="",
        category="",
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
    scaffold_chunks: tuple[Any, Any],
    first_sample_chunks: dict[str, Any],
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

    first_sample_bytes = sum(chunk_bytes(chunk) for chunk in first_sample_chunks.values())
    first_sample_tokens = sum(len(chunk.token_ids) for chunk in first_sample_chunks.values())
    if first_sample_tokens <= 0 or not first_sample_chunks:
        return
    bytes_per_token = first_sample_bytes / first_sample_tokens

    header_chunk, footer_chunk = scaffold_chunks
    scaffold_tokens = len(header_chunk.token_ids) + len(footer_chunk.token_ids)
    retrieved_facts = min(config.top_k, len(first_sample_chunks))
    average_fact_tokens = first_sample_tokens / len(first_sample_chunks)
    composed_tokens = scaffold_tokens + retrieved_facts * average_fact_tokens
    composed_bytes = total_requests * composed_tokens * bytes_per_token
    source_bytes = first_sample_bytes * unique_sample_count

    device_total = torch.cuda.get_device_properties(0).total_memory
    vllm_pool = config.kv_gpu_memory_utilization * device_total

    # Per-backend HBM-resident component: cpu-pinned holds composed memories
    # in pinned host RAM and only stages one copy per slot; the GPU store
    # keeps every composed copy resident.
    if config.kv_store_backend == "cpu-pinned":
        resident_label = "staging"
        resident_bytes = config.kv_staging_slots * composed_tokens * bytes_per_token
        remediation = "Shrink the user count, lower --top-k, --kv-staging-slots, or --gpu-memory-utilization."
    else:
        resident_label = "composed"
        resident_bytes = composed_bytes
        remediation = "Shrink the user count, lower --top-k, or reduce --gpu-memory-utilization."

    projected_peak = vllm_pool + resident_bytes + source_bytes
    if projected_peak > 0.97 * device_total:
        raise RuntimeError(
            "Projected KV GPU footprint exceeds device memory: "
            f"vllm_pool={vllm_pool / 2**30:.1f}GiB "
            f"{resident_label}={resident_bytes / 2**30:.1f}GiB "
            f"sources={source_bytes / 2**30:.1f}GiB "
            f"device={device_total / 2**30:.1f}GiB "
            f"(users={num_users}, requests={total_requests}). "
            f"{remediation}"
        )

    if config.kv_store_backend == "cpu-pinned":
        try:
            host_total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            return
        # Pinned allocations are non-swappable; keep a conservative margin.
        if composed_bytes > 0.8 * host_total:
            raise RuntimeError(
                "Projected pinned-host KV footprint exceeds 80% of system RAM: "
                f"composed={composed_bytes / 2**30:.1f}GiB "
                f"host={host_total / 2**30:.1f}GiB "
                f"(users={num_users}, requests={total_requests}). "
                "Shrink the user count or lower --top-k."
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


def _warm_up(
    llm: Any,
    prompts: list[dict[str, list[int]]],
    sampling_params: Any,
    batches: int,
    *,
    seed: int,
    kv_warmup: tuple[dict[str, list[int]], Any] | None = None,
) -> None:
    """Warm up with random tokens so no real prompt lands in the prefix cache.

    Prefix caching is enabled, so warming up with the measured prompts would
    let the measured batch hit the cache (inflated QPS) and, worse, let the
    native cache satisfy memory prefixes so the connector never injects KV.
    Warmup only needs to exercise kernels, allocator, and scheduler; token
    identity is irrelevant. kv_warmup is one (prompt, sampling_params) pair
    carrying explicit memory routing so the injection path warms too.
    """
    if batches > 0 and prompts:
        vocab_size = _tokenizer_vocab_size(llm.get_tokenizer())
        warmup_prompts = _build_warmup_prompts(prompts, vocab_size=vocab_size, seed=seed)
        warmup_sampling: Any = sampling_params
        if kv_warmup is not None:
            kv_prompt, kv_sampling = kv_warmup
            warmup_sampling = [*([sampling_params] * len(warmup_prompts)), kv_sampling]
            warmup_prompts = [*warmup_prompts, kv_prompt]
        for _ in range(batches):
            llm.generate(warmup_prompts, warmup_sampling, use_tqdm=False)
    # Reset even with warmup disabled: conditions that measure several user
    # counts against one engine must not let count N's prompts serve count
    # N+1 from the native cache.
    _reset_prefix_cache(llm)


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    return int(getattr(tokenizer, "vocab_size", 0) or 0) or len(tokenizer)


def _sampling_params_with_memory_user_ids(
    sampling_params: Any,
    memory_user_ids: list[str],
) -> list[Any]:
    """Clone the base params once per request, routing each to its memory."""
    clone = getattr(sampling_params, "clone", None)
    if not callable(clone):
        raise RuntimeError(
            "Pinned vLLM SamplingParams must provide clone() for request routing."
        )
    routed: list[Any] = []
    for memory_user_id in memory_user_ids:
        request_params = clone()
        extra_args = dict(getattr(request_params, "extra_args", None) or {})
        extra_args[MEMORY_USER_ID_EXTRA_ARG] = memory_user_id
        request_params.extra_args = extra_args
        routed.append(request_params)
    return routed


# Mixed-script probe so tokenizer stacks that differ in template, whitespace,
# byte-fallback, or unicode handling cannot encode it identically by accident.
_TOKENIZER_PARITY_PROBE_TEXT = (
    "SPEAKER Caroline (2023-05-08): I'll re-check trip #42 - cost $1,300.50; "
    "email caroline@example.com, emoji \U0001f642, CJK 你好, newline\nend.\n"
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


def _build_kv_warmup_prompt(
    memory_token_ids: list[int],
    *,
    vocab_size: int,
    seed: int,
) -> dict[str, list[int]]:
    """One warmup prompt whose prefix IS a registered memory.

    Random warmup prompts match no memory, so without this the connector's
    match/inject path (scatter kernels, staging) first runs inside the
    measured batch. The prompt must be strictly longer than the memory for
    the match to fire; the tail is seeded-random so it shares nothing with
    real queries.
    """
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1 to build the KV warmup prompt.")
    rng = random.Random(seed)
    tail = [rng.randrange(vocab_size) for _ in range(8)]
    return {"prompt_token_ids": [*memory_token_ids, *tail]}


def _build_warmup_prompts(
    prompts: list[dict[str, list[int]]],
    *,
    vocab_size: int,
    seed: int,
) -> list[dict[str, list[int]]]:
    if vocab_size < 1:
        raise ValueError("vocab_size must be >= 1 to build warmup prompts.")
    rng = random.Random(seed)
    return [
        {"prompt_token_ids": [rng.randrange(vocab_size) for _ in prompt["prompt_token_ids"]]}
        for prompt in prompts[: min(10, len(prompts))]
    ]


def _reset_prefix_cache(llm: Any) -> None:
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
    num_users: int,
    *,
    condition: str,
    vector_backend: str | None = None,
    jasper_effective_beam_width: int | None = None,
    fact_count: float = 0.0,
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
    kv_store_backend: str = "gpu",
    kv_store_host_mb: float = 0.0,
    kv_store_write_time_s: float = 0.0,
    kv_h2d_bytes: int = 0,
    kv_h2d_avg_ms: float = 0.0,
    kv_h2d_p95_ms: float = 0.0,
    kv_h2d_overlap_ratio: float = 0.0,
    kv_staging_stall_ms: float = 0.0,
    kv_requests_loaded: int = 0,
    total_input_tokens: int,
    total_output_tokens: int,
) -> dict[str, Any]:
    total_requests = num_users * config.requests_per_user
    if generation_time_s <= 0:
        raise RuntimeError("Measured benchmark time must be greater than zero.")
    row = {
        "run_id": config.run_id,
        "model": config.model,
        "model_label": config.model_label,
        "condition": condition,
        "vector_backend": vector_backend,
        "jasper_effective_beam_width": jasper_effective_beam_width,
        "num_users": num_users,
        "fact_count": fact_count,
        "requests_per_user": config.requests_per_user,
        "total_requests": total_requests,
        "throughput_qps": total_requests / generation_time_s,
        "avg_latency_ms": generation_time_s / total_requests * 1000,
        "generation_time_s": generation_time_s,
        "retrieval_time_s": retrieval_time_s,
        "vector_search_time_s": vector_search_time_s,
        "prompt_build_time_s": prompt_build_time_s,
        "kv_compose_time_s": kv_compose_time_s,
        "kv_verify_time_s": kv_verify_time_s,
        "memory_setup_time_s": memory_setup_time_s,
        "kv_precompute_time_s": kv_precompute_time_s,
        "engine_startup_time_s": engine_startup_time_s,
        "kv_prefix_caching": int(config.kv_enable_prefix_caching),
        "kv_store_gpu_mb": kv_store_gpu_mb,
        "kv_store_backend": kv_store_backend,
        "kv_store_host_mb": kv_store_host_mb,
        "kv_store_write_time_s": kv_store_write_time_s,
        "kv_h2d_bytes": kv_h2d_bytes,
        "kv_h2d_avg_ms": kv_h2d_avg_ms,
        "kv_h2d_p95_ms": kv_h2d_p95_ms,
        "kv_h2d_overlap_ratio": kv_h2d_overlap_ratio,
        "kv_staging_stall_ms": kv_staging_stall_ms,
        "kv_requests_loaded": kv_requests_loaded,
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
