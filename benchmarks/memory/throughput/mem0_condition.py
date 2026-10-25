"""The mem0 prompt-injection throughput conditions: retrieved facts injected as prompt text.

mem0_qdrant and mem0_jasper differ only in the vector backend of the per-sample
store. Every user's requests are searched once, and a request's prompt and
timings are charged to every user count whose batch includes that user, so one
worker measures all user counts against the same engine.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from loguru import logger

from benchmarks.memory.data import ConversationSample
from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.prompting import (
    build_memory_prompt_token_ids,
    extract_memory_scaffold_token_ids,
)
from benchmarks.memory.throughput.batches import (
    PreparedRequest,
    UserCountAccumulator,
    prepare_user_batches,
)
from benchmarks.memory.throughput.config import ThroughputConfig, jasper_effective_beam_width
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


@dataclass(slots=True)
class _ConditionContext:
    config: ThroughputConfig
    condition: str
    tokenizer: Any
    scaffold: ScaffoldTokens
    samples: Sequence[ConversationSample]
    stores: SampleStores
    facts_by_sample: Mapping[str, tuple[MemoryFact, ...]]


def run_mem0(
    config: ThroughputConfig, user_counts: tuple[int, ...], *, condition: str, backend: str
) -> list[dict[str, Any]]:
    require_embedding_access(config, condition)
    samples = load_samples(config)
    catalog_store = fact_catalog_store(config)
    max_users = max(user_counts)
    requests = build_locomo_requests(
        samples, num_users=max_users, requests_per_user=config.requests_per_user, seed=config.seed
    )
    used_samples = samples_for_users(samples, max_users)
    facts_by_sample = {sample.sample_id: catalog_store.load(sample) for sample in used_samples}
    store_root = config.local_store_scratch_dir / f"{condition}-stores"
    shutil.rmtree(store_root, ignore_errors=True)
    stores: SampleStores | None = None
    engine: StartedEngine | None = None
    try:
        engine = start_llm(config)
        tokenizer = engine.llm.get_tokenizer()
        logger.info(
            "{}: preparing {} sample stores for {} users", condition, len(used_samples), max_users
        )
        stores = prepare_sample_stores(
            config,
            used_samples,
            backend=backend,
            store_root=store_root,
            facts_by_sample=facts_by_sample,
        )
        context = _ConditionContext(
            config=config,
            condition=condition,
            tokenizer=tokenizer,
            scaffold=extract_memory_scaffold_token_ids(tokenizer, block_size=config.kv_block_size),
            samples=samples,
            stores=stores,
            facts_by_sample=facts_by_sample,
        )
        batches = prepare_user_batches(
            condition=condition,
            samples=samples,
            stores=stores,
            facts_by_sample=facts_by_sample,
            user_counts=user_counts,
            requests_by_user=group_requests_by_user(requests, max_users),
            prepare_request=partial(_prepare_request, context),
        )
        return [_measure(context, backend, engine, batch) for batch in batches]
    finally:
        if stores is not None:
            stores.close()
        shutil.rmtree(store_root, ignore_errors=True)
        release_llm(engine.llm if engine is not None else None)


def _prepare_request(
    context: _ConditionContext, sample: ConversationSample, request: LocomoRequest
) -> PreparedRequest:
    memory = context.stores.memory(sample.sample_id)
    search = search_store(memory, request.query, top_k=context.config.top_k)
    started = time.perf_counter()
    memory_prompt = build_memory_prompt_token_ids(
        context.tokenizer, sample, search.result, context_window=0, memory_scaffold=context.scaffold
    )
    prompt_token_ids = request_prompt_token_ids(context.tokenizer, memory_prompt.token_ids, request)
    return PreparedRequest(
        prompt_token_ids=prompt_token_ids,
        retrieval_s=search.elapsed_s,
        search_s=search.search_s,
        prompt_build_s=time.perf_counter() - started,
    )


def _measure(
    context: _ConditionContext, backend: str, engine: StartedEngine, batch: UserCountAccumulator
) -> dict[str, Any]:
    config = context.config
    logger.info("{}: users={}", context.condition, batch.num_users)
    prepare_batch(config, engine.llm, batch.prompts, engine.sampling_params)
    measured = measure_batch(engine.llm, batch.prompts, engine.sampling_params)
    return build_result_row(
        config,
        batch.num_users,
        condition=context.condition,
        vector_backend=backend,
        jasper_effective_beam_width=(
            jasper_effective_beam_width(config) if backend == "jasper" else None
        ),
        fact_count=batch.fact_count_total / batch.num_users,
        generation_time_s=float(measured["generation_time_s"]),
        retrieval_time_s=batch.retrieval_time_s,
        vector_search_time_s=batch.vector_search_time_s,
        prompt_build_time_s=batch.prompt_build_time_s,
        memory_setup_time_s=batch.memory_setup_time_s,
        engine_startup_time_s=engine.startup_time_s,
        total_input_tokens=int(measured["total_input_tokens"]),
        total_output_tokens=int(measured["total_output_tokens"]),
    )
