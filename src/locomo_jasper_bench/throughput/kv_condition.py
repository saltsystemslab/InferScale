"""The kv_injection throughput condition: retrieve, compose, and inject KV."""

from __future__ import annotations

import shutil
import time
import uuid
from typing import Any

from ..kv.chunk_cache import (
    CachedSampleEncode,
    cache_meta,
    cache_path_for,
    load_sample_chunks,
    save_sample_chunks,
    scaffold_chunks_match,
)
from ..kv.chunk_store import (
    build_chunk_store,
    close_chunk_store,
    fetch_fact_chunks,
    register_fact_chunks,
    release_fact_chunks,
)
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
from ..kv.tokenization import encode_text_no_special
from ..retrieval.fact_catalog import fact_catalog_hits
from ..runtime_paths import local_store_scratch_dir
from .config import ThroughputConfig
from .engine import (
    build_kv_warmup_prompt,
    measure_batch,
    release_llm,
    sampling_params_with_memory_user_ids,
    start_llm,
    tokenizer_vocab_size,
    validate_prompt_lengths,
    warm_up,
)
from .projection import check_kv_gpu_projection
from .reporting import build_result_row
from .stores import (
    build_user_store,
    close_mem0,
    fact_catalog_store,
    load_samples,
    search_store,
)
from .workload import LocomoRequest, build_locomo_requests, request_question_answer


def run_kv_injection(config: ThroughputConfig, num_users: int) -> dict[str, Any]:
    """Jasper retrieval per request, injecting the retrieved facts' pre-encoded KV.

    This is the head-to-head counterpart of mem0_jasper: identical per-user
    stores and identical top-k searches on the critical path, but retrieved
    memories reach the model as composed KV tensors instead of prompt text.

    Runs as ordered phases - encode, retrieve+compose, generate - and frees
    the encoder weights and the fact-chunk corpus store before the engine
    allocates its pool; the jasper stores stay resident for the whole run
    (segments are small on the current jasperpy branch) until the finally
    block. The corpus lives in the --kv-store-backend store (HBM or pinned
    host RAM); composed request memories are ephemeral GPU products held
    only in the connector's in-flight registry.
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

    from ..kv.chunked_rope import ChunkedRopeEncoder, load_encoder_tokenizer
    from ..kv.gpu_registry import (
        drop_namespace,
        get_gpu_memory_store,
        namespace_stats,
        register_user_memory,
    )
    from ..kv.vllm_runtime import build_strict_gpu_kv_transfer_config, force_vllm_inprocess_mode

    connector_module = importlib.import_module(config.kv_connector_module)
    force_vllm_inprocess_mode()
    samples = load_samples(config)
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
    catalog_store = fact_catalog_store(config)
    facts_by_sample = {
        sample.sample_id: catalog_store.load(sample) for sample in used_samples
    }

    namespace = f"throughput-{config.run_id}-{uuid.uuid4().hex}"
    # The namespace registry holds in-flight per-request compositions for the
    # connector handoff and is always GPU-resident; it is NOT the memory
    # store. --kv-store-backend selects where the fact-chunk corpus lives
    # (the chunk store built below).
    get_gpu_memory_store(namespace, backend="gpu")
    mem0_store_root = local_store_scratch_dir(config.run_id) / f"kv-stores-{num_users}u"
    if mem0_store_root.exists():
        shutil.rmtree(mem0_store_root)
    encoder: Any | None = None
    llm: Any | None = None
    chunk_store: Any | None = None
    stores_by_sample: dict[str, Any] = {}
    try:
        print(f"kv_injection: users={num_users} precompute starting", flush=True)
        precompute_started = time.perf_counter()
        tokenizer = load_encoder_tokenizer(config.model)
        scaffold = extract_memory_scaffold_token_ids(
            tokenizer,
            used_samples[0],
            block_size=config.kv_block_size,
        )

        kv_facts_by_sample = {
            sample.sample_id: unique_memory_facts(
                fact_catalog_hits(facts_by_sample[sample.sample_id])
            )
            for sample in used_samples
        }

        def sample_cache_key(sample: Any) -> tuple[Any, dict[str, Any]]:
            key_kwargs = dict(
                model=config.model,
                dtype=config.kv_dtype,
                context_window=config.context_window,
                max_position=config.kv_max_position,
                block_size=config.kv_block_size,
                sample=sample,
                facts=kv_facts_by_sample[sample.sample_id],
            )
            return cache_path_for(**key_kwargs), cache_meta(**key_kwargs)

        # Fact chunks head into the backend chunk store: under the cpu
        # backend they load from disk straight to host and get pinned by the
        # store, never touching the GPU; scaffold chunks and RoPE tables are
        # compose-side and always land on the device.
        chunk_device = "cpu" if config.kv_store_backend == "cpu" else config.kv_device
        cached_by_sample: dict[str, CachedSampleEncode] = {}
        if config.kv_chunk_cache_enabled:
            for sample in used_samples:
                cache_path, payload_meta = sample_cache_key(sample)
                cached = load_sample_chunks(
                    cache_path,
                    device=chunk_device,
                    scaffold_device=config.kv_device,
                    expected_meta=payload_meta,
                    expected_fact_ids=[
                        fact.memory_id for fact in kv_facts_by_sample[sample.sample_id]
                    ],
                )
                if cached is not None and scaffold_chunks_match(cached.scaffold_chunks, scaffold):
                    cached_by_sample[sample.sample_id] = cached

        if all(sample.sample_id in cached_by_sample for sample in used_samples):
            # Every sample came from the chunk cache: build the encoder from
            # the cached RoPE tables and never load the HF weights.
            first_cached = cached_by_sample[used_samples[0].sample_id]
            encoder = ChunkedRopeEncoder.from_tables(
                model=config.model,
                device=config.kv_device,
                max_position=config.kv_max_position,
                tokenizer=tokenizer,
                cos_table=first_cached.cos_table,
                sin_table=first_cached.sin_table,
            )
            header_chunk = first_cached.scaffold_chunks["header"]
            footer_chunk = first_cached.scaffold_chunks["footer"]
            memory_list_header_chunk = first_cached.scaffold_chunks.get("memory_list_header")
            empty_memory_chunk = first_cached.scaffold_chunks.get("empty_memory")
        else:
            encoder = ChunkedRopeEncoder(
                model=config.model,
                dtype=config.kv_dtype,
                device=config.kv_device,
                max_position=config.kv_max_position,
            )
            tokenizer = encoder.tokenizer
            header_chunk = encoder.encode_token_ids_chunk("scaffold:header", scaffold.header_token_ids)
            footer_chunk = encoder.encode_token_ids_chunk("scaffold:footer", scaffold.footer_token_ids)
            # The condition composes with header/footer only, but the cache
            # payload is shared with the accuracy composer, which needs all
            # four scaffold slots.
            memory_list_header_chunk = (
                encoder.encode_token_ids_chunk(
                    "scaffold:memory_list_header", scaffold.memory_list_header_token_ids
                )
                if scaffold.memory_list_header_token_ids
                else None
            )
            empty_memory_chunk = encoder.encode_token_ids_chunk(
                "scaffold:empty_memory", scaffold.empty_memory_token_ids
            )

        chunk_store = build_chunk_store(
            config.kv_store_backend,
            device=config.kv_device,
            top_k=config.top_k,
            staging_slots=config.kv_staging_slots,
        )
        # Values are metadata-only chunk maps once registration moves each
        # sample's KV into the chunk store below.
        chunks_by_sample: dict[str, dict[str, Any]] = {}
        for sample_number, sample in enumerate(used_samples, start=1):
            cached = cached_by_sample.get(sample.sample_id)
            if cached is not None:
                fact_chunks = cached.fact_chunks
                print(
                    f"kv_injection: loaded sample {sample_number}/{len(used_samples)} "
                    f"from chunk cache ({len(fact_chunks)} facts)",
                    flush=True,
                )
            else:
                kv_facts = kv_facts_by_sample[sample.sample_id]
                fact_token_ids = {
                    fact.memory_id: encode_text_no_special(tokenizer, format_memory_fact(fact))
                    for fact in kv_facts
                }
                turn_token_ids: dict[str, list[int]] = {}
                fact_chunks = {}
                for fact in kv_facts:
                    # Chunk values are conditioned on the preceding context_window
                    # turns; only the fact-token KV slice is kept (prefix-discard),
                    # matching the accuracy path's vllm-kv encoding semantics.
                    plan = build_fact_context_encoding_plan(
                        fact,
                        sample,
                        tokenizer=tokenizer,
                        context_window=config.context_window,
                        max_input_tokens=config.kv_max_position,
                        fact_token_ids=fact_token_ids,
                        turn_token_ids=turn_token_ids,
                    )
                    fact_chunks[fact.memory_id] = encoder.encode_fact_chunk(plan)
                if config.kv_chunk_cache_enabled:
                    cache_path, payload_meta = sample_cache_key(sample)
                    try:
                        save_sample_chunks(
                            cache_path,
                            meta=payload_meta,
                            fact_chunks=fact_chunks,
                            scaffold_chunks={
                                "header": header_chunk,
                                "memory_list_header": memory_list_header_chunk,
                                "empty_memory": empty_memory_chunk,
                                "footer": footer_chunk,
                            },
                            cos_table=encoder.cos_table,
                            sin_table=encoder.sin_table,
                        )
                    except Exception as exc:
                        # The cache is an accelerator, never a correctness
                        # dependency; a full disk must not kill the run.
                        print(f"kv_injection: chunk cache save failed: {exc}", flush=True)
                print(
                    f"kv_injection: encoded sample {sample_number}/{len(used_samples)} "
                    f"({len(fact_chunks)} facts)",
                    flush=True,
                )
            if sample_number == 1:
                check_kv_gpu_projection(
                    config,
                    num_users=num_users,
                    total_requests=len(requests),
                    unique_sample_count=len(used_samples),
                    scaffold_chunks=(header_chunk, footer_chunk),
                    first_sample_chunks=fact_chunks,
                )
            # Move the corpus into the backend store; keep metadata-only
            # chunks locally for lookups and error messages.
            chunks_by_sample[sample.sample_id] = register_fact_chunks(chunk_store, fact_chunks)
        encoder_probe_token_ids = encode_text_no_special(
            tokenizer, _TOKENIZER_PARITY_PROBE_TEXT
        )
        encoder.release_model()
        torch.cuda.synchronize()
        # Return the released encoder weights to the driver: the jasper
        # graph builds ahead allocate outside torch's caching allocator.
        torch.cuda.empty_cache()
        kv_precompute_time_s = time.perf_counter() - precompute_started

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

        # Users round-robin over the same samples and a store's contents
        # depend only on the sample, so one store per sample serves all of
        # its users.
        setup_started = time.perf_counter()
        for sample in used_samples:
            stores_by_sample[sample.sample_id] = build_user_store(
                config,
                backend="jasper",
                store_root=mem0_store_root / sample.sample_id,
                facts=facts_by_sample[sample.sample_id],
            )
        memory_setup_time_s = time.perf_counter() - setup_started

        for user_index in range(num_users):
            sample = samples_by_user[user_index]
            facts = facts_by_sample[sample.sample_id]
            fact_count_total += len(facts)
            open_memory = stores_by_sample[sample.sample_id]
            for request in requests_by_user[user_index]:
                hits, elapsed_s, search_s = search_store(
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
                selected_ids = [fact.memory_id for fact in selected_facts]
                if not selected_ids:
                    raise RuntimeError("Retrieval returned no facts for a kv_injection request.")
                # Stages the selected chunks out of the corpus store (async
                # H2D under the cpu backend), so compose time carries the
                # PCIe cost of host-resident sources.
                selected = fetch_fact_chunks(
                    chunk_store, chunks_by_sample[sample.sample_id], selected_ids
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
                release_fact_chunks(chunk_store, selected_ids)

                # Registration hands the composed memory to the GPU-resident
                # in-flight registry (a dict insert, no transfer); timed for
                # parity with the other setup metrics.
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
                qa = request_question_answer(request)
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
            if (user_index + 1) % 10 == 0 or user_index + 1 == num_users:
                print(f"kv_injection: retrieved {user_index + 1}/{num_users} users", flush=True)

        # The corpus's only consumer is composition; capture its stats and
        # transfer metrics for the result row, then free it before the
        # engine claims its pool (the disk cache is the durable layer
        # across runs). The jasper stores stay open (a graph segment is
        # ~13.5MiB on the current jasperpy branch) until the finally block.
        chunk_store_stats = chunk_store.get_stats()
        chunk_bench_summary = chunk_store.get_bench_summary()
        close_chunk_store(chunk_store)
        chunks_by_sample.clear()
        cached_by_sample.clear()
        del header_chunk, footer_chunk, memory_list_header_chunk, empty_memory_chunk
        torch.cuda.empty_cache()

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
            # The connector's namespace registry holds in-flight composed
            # memories and is always GPU-resident; the corpus backend is the
            # chunk store's concern, not the connector's.
            store_backend="gpu",
        )
        llm, sampling_params, engine_startup_time_s = start_llm(
            config,
            kv_transfer_config=transfer_config,
        )
        _require_tokenizer_parity(encoder_probe_token_ids, llm.get_tokenizer())

        store_stats = namespace_stats(namespace)
        validate_prompt_lengths(config, prompts)
        routed_sampling_params = sampling_params_with_memory_user_ids(
            sampling_params,
            prompt_memory_user_ids,
        )
        # With prefix scan off, the KV warmup prompt reaches its memory via
        # the same explicit routing as measured requests; the random warmup
        # prompts stay unrouted so they match nothing.
        kv_warmup: tuple[dict[str, list[int]], Any] | None = None
        if first_memory_token_ids and first_memory_user_id:
            kv_warmup = (
                build_kv_warmup_prompt(
                    first_memory_token_ids,
                    vocab_size=tokenizer_vocab_size(tokenizer),
                    seed=config.seed,
                ),
                sampling_params_with_memory_user_ids(
                    sampling_params, [first_memory_user_id]
                )[0],
            )
        warm_up(
            llm,
            prompts,
            sampling_params,
            config.warmup_batches,
            seed=config.seed,
            kv_warmup=kv_warmup,
        )
        connector_module.reset_load_stats()
        measured = measure_batch(llm, prompts, routed_sampling_params)
        load_stats = connector_module.snapshot_load_stats()
        # Composed memories are GPU-resident in the registry under both
        # backends; the corpus footprint and its H2D transfers were captured
        # from the chunk store before it closed.
        kv_store_gpu_mb = float(store_stats.get("total_gpu_mb", 0.0))
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
        return build_result_row(
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
            kv_store_host_mb=float(chunk_store_stats.get("total_host_mb", 0.0)),
            kv_store_write_time_s=kv_store_write_time_s,
            kv_h2d_bytes=int(chunk_bench_summary.get("total_bytes_transferred", 0)),
            kv_h2d_avg_ms=float(chunk_bench_summary.get("avg_h2d_latency_ms", 0.0)),
            kv_h2d_p95_ms=float(chunk_bench_summary.get("p95_h2d_latency_ms", 0.0)),
            kv_h2d_overlap_ratio=float(chunk_bench_summary.get("avg_overlap_ratio", 0.0)),
            kv_staging_stall_ms=float(chunk_bench_summary.get("total_staging_stall_ms", 0.0)),
            kv_requests_loaded=int(load_stats["requests_loaded"]),
            total_input_tokens=measured["total_input_tokens"],
            total_output_tokens=measured["total_output_tokens"],
        )
    finally:
        for store in stores_by_sample.values():
            close_mem0(store)
        if chunk_store is not None:
            close_chunk_store(chunk_store)
        release_llm(llm)
        if encoder is not None:
            encoder.close()
        drop_namespace(namespace)
        shutil.rmtree(mem0_store_root, ignore_errors=True)


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
