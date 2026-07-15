"""The mem0 prompt-injection throughput conditions (qdrant/jasper retrieval)."""

from __future__ import annotations

import shutil
import time
from typing import Any

from ..kv.prompting import (
    build_kv_equivalence_prompt_token_ids,
    build_memory_prompt_token_ids,
    extract_memory_scaffold_token_ids,
)
from ..retrieval.fact_catalog import MemoryFact
from ..runtime_paths import local_store_scratch_dir
from .config import ThroughputConfig
from .engine import measure_batch, release_llm, start_llm, validate_prompt_lengths, warm_up
from .reporting import build_result_row
from .stores import (
    build_user_store,
    close_mem0,
    fact_catalog_store,
    load_samples,
    search_store,
)
from .workload import LocomoRequest, build_locomo_requests, request_question_answer


def run_mem0(
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

    samples = load_samples(config)
    catalog_store = fact_catalog_store(config)
    facts_by_sample: dict[str, tuple[MemoryFact, ...]] = {}

    llm: Any | None = None
    stores_by_sample: dict[str, Any] = {}
    mem0_store_root = local_store_scratch_dir(config.run_id) / f"{condition}-stores"
    if mem0_store_root.exists():
        shutil.rmtree(mem0_store_root)
    try:
        llm, sampling_params, engine_startup_time_s = start_llm(config)
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

        # Users round-robin over the same samples and a store's contents
        # depend only on the sample, so one store per sample serves all of
        # its users; each request is searched once and its measured cost is
        # attributed to every count whose batch includes it.
        used_samples = [samples[index % len(samples)] for index in range(min(max_users, len(samples)))]
        print(
            f"{condition}: preparing {len(used_samples)} sample stores for {max_users} users",
            flush=True,
        )
        sample_store_build_s: dict[str, float] = {}
        for sample in used_samples:
            facts_by_sample[sample.sample_id] = catalog_store.load(sample)
            setup_started = time.perf_counter()
            stores_by_sample[sample.sample_id] = build_user_store(
                config,
                backend=backend,
                store_root=mem0_store_root / sample.sample_id,
                facts=facts_by_sample[sample.sample_id],
            )
            sample_store_build_s[sample.sample_id] = time.perf_counter() - setup_started
        for count in user_counts:
            accumulators[count]["memory_setup_time_s"] = sum(
                sample_store_build_s[samples[index % len(samples)].sample_id]
                for index in range(min(count, len(samples)))
            )

        for user_index in range(max_users):
            sample = samples[user_index % len(samples)]
            facts = facts_by_sample[sample.sample_id]
            open_memory = stores_by_sample[sample.sample_id]
            applicable_counts = [count for count in user_counts if user_index < count]
            for count in applicable_counts:
                accumulators[count]["fact_count_total"] += len(facts)
            for request in requests_by_user[user_index]:
                hits, elapsed_s, search_s = search_store(
                    open_memory,
                    request.query,
                    top_k=config.top_k,
                )

                prompt_started = time.perf_counter()
                memory_prompt = build_memory_prompt_token_ids(
                    tokenizer,
                    sample,
                    hits,
                    context_window=0,
                    memory_scaffold=scaffold,
                )
                qa = request_question_answer(request)
                prompt_token_ids = build_kv_equivalence_prompt_token_ids(
                    tokenizer,
                    memory_prompt.token_ids,
                    sample,
                    qa,
                    memory_scaffold=scaffold,
                ).prompt_token_ids
                prompt_build_s = time.perf_counter() - prompt_started

                for count in applicable_counts:
                    accumulator = accumulators[count]
                    accumulator["retrieval_time_s"] += elapsed_s
                    accumulator["vector_search_time_s"] += search_s
                    accumulator["prompts"].append({"prompt_token_ids": prompt_token_ids})
                    accumulator["prompt_build_time_s"] += prompt_build_s
            if (user_index + 1) % 10 == 0 or user_index + 1 == max_users:
                print(f"{condition}: prepared {user_index + 1}/{max_users} users", flush=True)

        for count in user_counts:
            print(f"{condition}: users={count}", flush=True)
            accumulator = accumulators[count]
            prompts = accumulator["prompts"]
            validate_prompt_lengths(config, prompts)
            warm_up(llm, prompts, sampling_params, config.warmup_batches, seed=config.seed)
            measured = measure_batch(llm, prompts, sampling_params)
            results.append(
                build_result_row(
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
        for store in stores_by_sample.values():
            close_mem0(store)
        shutil.rmtree(mem0_store_root, ignore_errors=True)
        release_llm(llm)
