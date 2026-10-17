"""The kv_injection throughput condition: Jasper retrieval per request with the facts' KV injected.

This is the head-to-head counterpart of mem0_jasper: identical per-user stores
and identical top-k searches on the critical path, but retrieved memories
reach the model as composed KV tensors instead of prompt text. The condition
runs as ordered phases: encode the corpus, build the retrieval stores, compose
every request, free the corpus, then start vLLM and measure one batch. The
corpus lives in the chunk store selected by ``inferscale.kv.store_backend``
(HBM or pinned host RAM); composed request memories are GPU-resident entries
of the connector's namespace registry until the batch finishes.
"""

from __future__ import annotations

import importlib
import shutil
import uuid
from types import ModuleType
from typing import Any

from benchmarks.memory.prompting import extract_memory_scaffold_token_ids
from benchmarks.memory.throughput.config import ThroughputConfig, jasper_effective_beam_width
from benchmarks.memory.throughput.engine import (
    KVWarmup,
    StartedEngine,
    build_kv_warmup_prompt,
    measure_batch,
    prepare_batch,
    release_llm,
    sampling_params_with_memory_user_ids,
    start_llm,
    tokenizer_vocab_size,
)
from benchmarks.memory.throughput.kv_corpus import CorpusStats, PrecomputedCorpus, precompute_corpus
from benchmarks.memory.throughput.kv_requests import (
    ComposedRequests,
    RequestContext,
    build_sample_retrieval,
    compose_requests,
)
from benchmarks.memory.throughput.projection import check_kv_gpu_projection
from benchmarks.memory.throughput.reporting import build_result_row
from benchmarks.memory.throughput.stores import (
    fact_catalog_store,
    load_samples,
    prepare_sample_stores,
    require_embedding_access,
)
from benchmarks.memory.throughput.workload import (
    build_locomo_requests,
    group_requests_by_user,
    samples_for_users,
)
from inferscale.v1.kv.encoder import require_tokenizer_parity
from inferscale.v1.kv.registry import drop_namespace, get_gpu_memory_store, namespace_stats
from inferscale.v1.serving.vllm import build_kv_transfer_config, force_vllm_inprocess_mode

CONDITION = "kv_injection"
GPU_DEVICES = {"cuda", "cuda:0"}


def run_kv_injection(config: ThroughputConfig, num_users: int) -> dict[str, Any]:
    _require_kv_settings(config)
    connector = load_connector_module(config)
    force_vllm_inprocess_mode()
    samples = load_samples(config)
    requests = build_locomo_requests(
        samples, num_users=num_users, requests_per_user=config.requests_per_user, seed=config.seed
    )
    used_samples = samples_for_users(samples, num_users)
    catalog_store = fact_catalog_store(config)
    facts_by_sample = {sample.sample_id: catalog_store.load(sample) for sample in used_samples}
    namespace = f"throughput-{config.run_id}-{uuid.uuid4().hex}"
    # The namespace registry holds the in-flight per-request compositions for
    # the connector handoff and is always GPU-resident; it is not the corpus
    # store, whose backend is inferscale.kv.store_backend.
    get_gpu_memory_store(namespace, backend="gpu")
    store_root = config.local_store_scratch_dir / f"kv-stores-{num_users}u"
    shutil.rmtree(store_root, ignore_errors=True)
    corpus: PrecomputedCorpus | None = None
    stores = None
    engine: StartedEngine | None = None
    try:
        corpus = precompute_corpus(config, used_samples, facts_by_sample)
        check_kv_gpu_projection(
            config,
            num_users=num_users,
            total_requests=len(requests),
            unique_sample_count=len(used_samples),
            footprint=corpus.footprint,
        )
        stores = prepare_sample_stores(
            config,
            used_samples,
            backend="jasper",
            store_root=store_root,
            facts_by_sample=facts_by_sample,
        )
        retrieval, row_map_time_s = build_sample_retrieval(corpus, stores)
        memory_setup_time_s = stores.setup_time_for_users(samples, num_users) + row_map_time_s
        context = RequestContext(
            config=config,
            corpus=corpus,
            retrieval=retrieval,
            namespace=namespace,
            scaffold=extract_memory_scaffold_token_ids(
                corpus.tokenizer, block_size=config.kv_block_size
            ),
        )
        composed = compose_requests(
            context, group_requests_by_user(requests, num_users), samples, facts_by_sample
        )
        # Composition was the corpus's only consumer: capture its statistics,
        # then free it before the engine claims its memory pool. The row maps
        # reference the chunk stores, so they go first.
        corpus_stats = corpus.stats()
        retrieval.clear()
        corpus.close()
        engine = start_kv_engine(config, namespace)
        require_tokenizer_parity(corpus.probe_token_ids, engine.llm.get_tokenizer())
        measured, load_stats = measure_kv_batch(
            config, engine, composed, corpus.tokenizer, connector
        )
        return _result_row(
            config,
            num_users,
            composed=composed,
            measured=measured,
            load_stats=load_stats,
            corpus=corpus,
            corpus_stats=corpus_stats,
            memory_setup_time_s=memory_setup_time_s,
            engine_startup_time_s=engine.startup_time_s,
            kv_store_gpu_mb=float(namespace_stats(namespace).get("total_gpu_mb", 0.0)),
        )
    finally:
        if stores is not None:
            stores.close()
        if corpus is not None:
            corpus.close()
        release_llm(engine.llm if engine is not None else None)
        drop_namespace(namespace)
        shutil.rmtree(store_root, ignore_errors=True)


def load_connector_module(config: ThroughputConfig) -> ModuleType:
    """The KV connector module, imported up front so a missing vLLM fails before any work."""
    return importlib.import_module(config.kv_connector_module)


def start_kv_engine(config: ThroughputConfig, namespace: str) -> StartedEngine:
    transfer_config = build_kv_transfer_config(
        connector_module=config.kv_connector_module,
        namespace=namespace,
        default_user_id=None,
        log_memory_hits=False,
        # The connector's namespace registry holds the in-flight composed
        # memories and is always GPU-resident; the corpus backend is the
        # chunk store's concern, not the connector's.
        store_backend="gpu",
    )
    return start_llm(config, kv_transfer_config=transfer_config)


def measure_kv_batch(
    config: ThroughputConfig,
    engine: StartedEngine,
    composed: ComposedRequests,
    tokenizer: Any,
    connector: ModuleType,
) -> tuple[dict[str, int | float], dict[str, int]]:
    """Warm up with explicit memory routing, then measure the batch and its KV coverage."""
    prepare_batch(
        config,
        engine.llm,
        composed.prompts,
        engine.sampling_params,
        kv_warmup=_kv_warmup(config, engine, composed, tokenizer),
    )
    routed = sampling_params_with_memory_user_ids(engine.sampling_params, composed.memory_user_ids)
    connector.reset_load_stats()
    measured = measure_batch(engine.llm, composed.prompts, routed)
    load_stats = connector.snapshot_load_stats()
    _require_full_coverage(load_stats, len(composed.prompts))
    return measured, load_stats


def _kv_warmup(
    config: ThroughputConfig, engine: StartedEngine, composed: ComposedRequests, tokenizer: Any
) -> KVWarmup | None:
    # The KV warmup prompt reaches its memory through the same explicit routing
    # as measured requests; the random warmup prompts stay unrouted so they
    # match nothing.
    if composed.first_memory_user_id is None or composed.first_memory_token_ids is None:
        return None
    prompt = build_kv_warmup_prompt(
        composed.first_memory_token_ids,
        vocab_size=tokenizer_vocab_size(tokenizer),
        seed=config.seed,
    )
    (sampling_params,) = sampling_params_with_memory_user_ids(
        engine.sampling_params, [composed.first_memory_user_id]
    )
    return KVWarmup(prompt=prompt, sampling_params=sampling_params)


def _require_full_coverage(load_stats: dict[str, int], prompt_count: int) -> None:
    # requests_covered counts distinct request ids that either had memory
    # injected or whose memory region was fully served by the native prefix
    # cache (a legitimate no-load), so the check stays exact per request even
    # with prefix caching enabled and preemption re-loads.
    if load_stats["requests_covered"] < prompt_count:
        raise RuntimeError(
            f"KV memory covered {load_stats['requests_covered']} of {prompt_count} requests "
            "(injected or natively cached); some requests generated without their memory."
        )


def _require_kv_settings(config: ThroughputConfig) -> None:
    if config.kv_device not in GPU_DEVICES:
        raise RuntimeError(
            "The connector namespace registry is GPU-resident; "
            "set inferscale.kv.device to cuda:0 in the run JSON."
        )
    require_embedding_access(config, CONDITION)


def _result_row(
    config: ThroughputConfig,
    num_users: int,
    *,
    composed: ComposedRequests,
    measured: dict[str, int | float],
    load_stats: dict[str, int],
    corpus: PrecomputedCorpus,
    corpus_stats: CorpusStats,
    memory_setup_time_s: float,
    engine_startup_time_s: float,
    kv_store_gpu_mb: float,
) -> dict[str, Any]:
    timings = composed.timings
    return build_result_row(
        config,
        num_users,
        condition=CONDITION,
        vector_backend="jasper",
        jasper_effective_beam_width=jasper_effective_beam_width(config),
        fact_count=composed.fact_count_total / num_users,
        generation_time_s=float(measured["generation_time_s"]),
        retrieval_time_s=timings.retrieval_s,
        vector_search_time_s=timings.vector_search_s,
        prompt_build_time_s=timings.prompt_build_s,
        memory_setup_time_s=memory_setup_time_s,
        kv_precompute_time_s=corpus.precompute_time_s,
        kv_compose_time_s=timings.compose_s,
        kv_verify_time_s=timings.verify_s,
        engine_startup_time_s=engine_startup_time_s,
        kv_store_gpu_mb=kv_store_gpu_mb,
        kv_store_backend=config.kv_store_backend,
        kv_store_host_mb=corpus_stats.host_mb,
        kv_store_write_time_s=timings.store_write_s,
        kv_h2d_bytes=corpus_stats.h2d_bytes,
        kv_h2d_avg_ms=corpus_stats.h2d_avg_ms,
        kv_h2d_p95_ms=corpus_stats.h2d_p95_ms,
        kv_h2d_overlap_ratio=corpus_stats.h2d_overlap_ratio,
        kv_staging_stall_ms=corpus_stats.staging_stall_ms,
        kv_requests_loaded=int(load_stats["requests_loaded"]),
        total_input_tokens=int(measured["total_input_tokens"]),
        total_output_tokens=int(measured["total_output_tokens"]),
    )
