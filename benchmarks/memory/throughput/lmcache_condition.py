"""The lmcache_blend throughput condition: retrieved Mem0 facts served through LMCache CacheBlend.

The condition retrieves like mem0_qdrant and sends the same prompt with
LMCache's separator between the facts, so each fact's KV, stored once by a
warm-up, is blended into the prefill instead of recomputed. Result rows reuse
the schema's KV columns with these meanings:

- ``kv_store_backend`` is ``lmcache-cpu``: the facts' KV lives in LMCache's pinned CPU cache.
- ``kv_store_host_mb`` is that cache's usage after the fact warm-up.
- ``kv_precompute_time_s`` is the fact warm-up time, excluded from QPS.
- ``kv_h2d_bytes`` is the KV LMCache matched during the measured batch, and
  ``kv_h2d_avg_ms`` its load-plus-blend time per request.
- ``kv_requests_loaded`` counts requests whose header and retrieved facts were
  all served from the cache; the condition fails unless that is every request.
- ``kv_prefix_caching`` is 0: vLLM's own prefix cache is off for this arm.

Every request also stores its unique footer-plus-question tail, which is
CacheBlend's genuine serving cost and stays inside the measured time.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from loguru import logger

from benchmarks.common.lmcache.config import (
    LMCacheConfig,
    engine_config_for_lmcache,
    lmcache_kv_transfer_config,
)
from benchmarks.common.lmcache.load_stats import reset_load_stats, snapshot_load_stats
from benchmarks.common.lmcache.runtime import (
    clear_cache,
    cpu_cache_usage_bytes,
    kv_bytes_per_token,
    register_model,
    require_cache_capacity,
    require_separator_parity,
    shutdown,
    warm_prompts,
)
from benchmarks.common.lmcache.segments import (
    blend_memory_layout,
    prompt_token_count,
    warm_up_prompts,
)
from benchmarks.memory.data import ConversationSample
from benchmarks.memory.mem0.fact_catalog import MemoryFact, fact_catalog_hits
from benchmarks.memory.prompting import extract_memory_scaffold_token_ids, select_memory_facts
from benchmarks.memory.throughput.batches import (
    PreparedRequest,
    UserCountAccumulator,
    prepare_user_batches,
)
from benchmarks.memory.throughput.config import ThroughputConfig, lmcache_config
from benchmarks.memory.throughput.engine import (
    StartedEngine,
    measure_batch,
    prepare_batch,
    release_llm,
    start_llm,
)
from benchmarks.memory.throughput.reporting import build_result_row
from benchmarks.memory.throughput.stores import (
    SampleStores,
    fact_catalog_store,
    load_samples,
    prepare_sample_stores,
    require_embedding_access,
    search_store,
)
from benchmarks.memory.throughput.workload import (
    LocomoRequest,
    build_locomo_requests,
    group_requests_by_user,
    request_prompt_token_ids,
    samples_for_users,
)
from inferscale.v1.kv.prompt import ScaffoldTokens
from inferscale.v1.units import bytes_to_mb

CONDITION = "lmcache_blend"
VECTOR_BACKEND = "qdrant"
STORE_BACKEND_LABEL = "lmcache-cpu"
WARM_BATCH_SIZE = 256
# The cache usage reported after the warm-up must reach this share of the
# facts' KV bytes, or the segments were not stored.
MIN_RESIDENT_FRACTION = 0.5


@dataclass(slots=True)
class _BlendContext:
    config: ThroughputConfig
    lmcache: LMCacheConfig
    tokenizer: Any
    scaffold: ScaffoldTokens
    separator: list[int]
    stores: SampleStores
    # The scaffold header and every catalog fact of the used samples, each its own prompt.
    warm_prompt_ids: list[list[int]]
    bytes_per_token: int


@dataclass(frozen=True, slots=True)
class FactCacheWarmup:
    """What priming LMCache with the facts cost and left resident."""

    time_s: float
    stored_tokens: int
    cpu_cache_bytes: int | None


def run_lmcache_blend(
    config: ThroughputConfig, user_counts: tuple[int, ...]
) -> list[dict[str, Any]]:
    lmcache = lmcache_config(config)
    require_embedding_access(config, CONDITION)
    samples = load_samples(config)
    catalog_store = fact_catalog_store(config)
    max_users = max(user_counts)
    requests = build_locomo_requests(
        samples, num_users=max_users, requests_per_user=config.requests_per_user, seed=config.seed
    )
    used_samples = samples_for_users(samples, max_users)
    facts_by_sample = {sample.sample_id: catalog_store.load(sample) for sample in used_samples}
    store_root = config.local_store_scratch_dir / f"{CONDITION}-stores"
    shutil.rmtree(store_root, ignore_errors=True)
    stores: SampleStores | None = None
    engine: StartedEngine | None = None
    try:
        engine = start_llm(
            config,
            kv_transfer_config=lmcache_kv_transfer_config(lmcache),
            engine=engine_config_for_lmcache(config.engine_config()),
        )
        register_model(engine.llm)
        tokenizer = engine.llm.get_tokenizer()
        separator = require_separator_parity(tokenizer, lmcache.blend_special_str)
        scaffold = extract_memory_scaffold_token_ids(tokenizer, block_size=config.kv_block_size)
        warm_prompt_ids = catalog_warm_up_prompts(
            tokenizer, scaffold, facts_by_sample.values(), separator
        )
        logger.info(
            "{}: preparing {} sample stores for {} users; {} warm-up prompts",
            CONDITION,
            len(used_samples),
            max_users,
            len(warm_prompt_ids),
        )
        stores = prepare_sample_stores(
            config,
            used_samples,
            backend=VECTOR_BACKEND,
            store_root=store_root,
            facts_by_sample=facts_by_sample,
        )
        context = _BlendContext(
            config=config,
            lmcache=lmcache,
            tokenizer=tokenizer,
            scaffold=scaffold,
            separator=separator,
            stores=stores,
            warm_prompt_ids=warm_prompt_ids,
            bytes_per_token=kv_bytes_per_token(),
        )
        batches = prepare_user_batches(
            condition=CONDITION,
            samples=samples,
            stores=stores,
            facts_by_sample=facts_by_sample,
            user_counts=user_counts,
            requests_by_user=group_requests_by_user(requests, max_users),
            prepare_request=partial(_prepare_request, context),
        )
        return [_measure(context, engine, batch) for batch in batches]
    finally:
        if stores is not None:
            stores.close()
        shutil.rmtree(store_root, ignore_errors=True)
        shutdown()
        release_llm(engine.llm if engine is not None else None)


def catalog_warm_up_prompts(
    tokenizer: Any,
    scaffold: ScaffoldTokens,
    catalogs: Iterable[Sequence[MemoryFact]],
    separator: Sequence[int],
) -> list[list[int]]:
    """The scaffold header and every catalog fact, tokenized exactly as the prompts lay them out."""
    chunks = [
        token_ids
        for facts in catalogs
        for token_ids in select_memory_facts(
            tokenizer, fact_catalog_hits(facts)
        ).ordered_token_ids()
    ]
    return warm_up_prompts(scaffold.header_token_ids, chunks, separator)


def _prepare_request(
    context: _BlendContext, sample: ConversationSample, request: LocomoRequest
) -> PreparedRequest:
    memory = context.stores.memory(sample.sample_id)
    search = search_store(memory, request.query, top_k=context.config.top_k)
    started = time.perf_counter()
    facts = select_memory_facts(context.tokenizer, search.result)
    layout = blend_memory_layout(context.scaffold, facts.ordered_token_ids(), context.separator)
    prompt_token_ids = request_prompt_token_ids(context.tokenizer, layout.token_ids, request)
    return PreparedRequest(
        prompt_token_ids=prompt_token_ids,
        retrieval_s=search.elapsed_s,
        search_s=search.search_s,
        prompt_build_s=time.perf_counter() - started,
        cached_prefix_tokens=layout.expected_matched_tokens,
    )


def _measure(
    context: _BlendContext, engine: StartedEngine, batch: UserCountAccumulator
) -> dict[str, Any]:
    config = context.config
    logger.info("{}: users={}", CONDITION, batch.num_users)
    # The random-token warm-up goes first: every request stores its segments,
    # and junk stored after the facts would be the last to be evicted.
    prepare_batch(config, engine.llm, batch.prompts, engine.sampling_params)
    tail_tokens = sum(
        len(prompt["prompt_token_ids"]) - cached
        for prompt, cached in zip(batch.prompts, batch.cached_prefix_tokens)
    )
    require_cache_capacity(
        warm_tokens=prompt_token_count(context.warm_prompt_ids),
        tail_tokens=tail_tokens,
        bytes_per_token=context.bytes_per_token,
        max_local_cpu_size_gb=context.lmcache.max_local_cpu_size_gb,
        what=f"{CONDITION} users={batch.num_users}",
    )
    warmup = prime_fact_cache(
        engine.llm, context.warm_prompt_ids, bytes_per_token=context.bytes_per_token
    )
    reset_load_stats()
    measured = measure_batch(engine.llm, batch.prompts, engine.sampling_params)
    stats = snapshot_load_stats()
    covered = count_covered_requests(stats["matched_tokens"], batch.cached_prefix_tokens)
    require_full_coverage(
        covered, len(batch.prompts), stats["matched_tokens"], batch.cached_prefix_tokens
    )
    logger.info(
        "{}: users={} warm_prompts={} cache_mb={:.1f} matched_tokens={} load_ms={:.1f} "
        "covered={}/{}",
        CONDITION,
        batch.num_users,
        len(context.warm_prompt_ids),
        bytes_to_mb(warmup.cpu_cache_bytes or 0),
        stats["tokens_matched"],
        stats["load_time_ms"],
        covered,
        len(batch.prompts),
    )
    return build_result_row(
        config,
        batch.num_users,
        condition=CONDITION,
        vector_backend=VECTOR_BACKEND,
        fact_count=batch.fact_count_total / batch.num_users,
        generation_time_s=float(measured["generation_time_s"]),
        retrieval_time_s=batch.retrieval_time_s,
        vector_search_time_s=batch.vector_search_time_s,
        prompt_build_time_s=batch.prompt_build_time_s,
        memory_setup_time_s=batch.memory_setup_time_s,
        kv_precompute_time_s=warmup.time_s,
        engine_startup_time_s=engine.startup_time_s,
        kv_store_backend=STORE_BACKEND_LABEL,
        kv_store_host_mb=bytes_to_mb(warmup.cpu_cache_bytes or 0),
        kv_h2d_bytes=int(stats["tokens_matched"]) * context.bytes_per_token,
        kv_h2d_avg_ms=float(stats["load_time_ms"]) / len(batch.prompts),
        kv_requests_loaded=covered,
        prefix_caching=False,
        total_input_tokens=int(measured["total_input_tokens"]),
        total_output_tokens=int(measured["total_output_tokens"]),
    )


def prime_fact_cache(
    llm: Any, prompts: Sequence[Sequence[int]], *, bytes_per_token: int
) -> FactCacheWarmup:
    """Empty LMCache, then store the header and every fact as its own segment."""
    clear_cache()
    stats = warm_prompts(llm, prompts, batch_size=WARM_BATCH_SIZE)
    usage = cpu_cache_usage_bytes()
    require_fact_cache_resident(usage, expected_bytes=stats.token_count * bytes_per_token)
    return FactCacheWarmup(
        time_s=stats.time_ms / 1000, stored_tokens=stats.token_count, cpu_cache_bytes=usage
    )


def require_fact_cache_resident(usage_bytes: int | None, *, expected_bytes: int) -> None:
    """After the warm-up the cache must hold the facts; a near-empty cache stored nothing."""
    if usage_bytes is None or expected_bytes <= 0:
        return
    if usage_bytes < MIN_RESIDENT_FRACTION * expected_bytes:
        raise RuntimeError(
            f"LMCache holds {usage_bytes} bytes after the fact warm-up, below half of the "
            f"expected {expected_bytes}; the fact segments were not stored. Check the separator "
            "parity and the LMCache logs."
        )


def count_covered_requests(matched_by_request: Mapping[str, int], expected: Sequence[int]) -> int:
    """Requests whose matched prefix covered the header and every retrieved fact.

    vLLM numbers the requests of one ``generate`` call consecutively, so the
    recorded request ids in numeric order line up with the batch's prompts.
    """
    if len(matched_by_request) != len(expected):
        raise RuntimeError(
            f"LMCache recorded {len(matched_by_request)} requests for a batch of "
            f"{len(expected)} prompts."
        )
    ordered = sorted(matched_by_request.items(), key=lambda item: _request_order(item[0]))
    return sum(1 for (_, matched), needed in zip(ordered, expected) if matched >= needed)


def require_full_coverage(
    covered: int, prompt_count: int, matched_by_request: Mapping[str, int], expected: Sequence[int]
) -> None:
    if covered >= prompt_count:
        return
    ordered = sorted(matched_by_request.items(), key=lambda item: _request_order(item[0]))
    shortfall = min(
        ((matched, needed) for (_, matched), needed in zip(ordered, expected) if matched < needed),
        key=lambda pair: pair[0] - pair[1],
    )
    raise RuntimeError(
        f"LMCache served the header and every retrieved fact for {covered} of {prompt_count} "
        f"requests (worst request matched {shortfall[0]} of {shortfall[1]} tokens); the rest were "
        "partly recomputed, so this batch does not measure CacheBlend."
    )


def _request_order(request_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(request_id))
    except ValueError:
        return (1, request_id)
