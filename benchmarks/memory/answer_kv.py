from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from typing import Any

from loguru import logger

from benchmarks.common.answer_timing import generation_latency_metrics
from benchmarks.common.clients import ChatResult
from inferscale.v1.kv.prompt import PromptTokens, build_prompt_tokens, require_identical_token_ids
from inferscale.v1.kv.registry import (
    clear_namespace,
    drop_namespace,
    get_gpu_memory_store,
    namespace_stats,
    register_user_memory,
    remove_user_memory,
)
from inferscale.v1.serving.vllm import (
    VLLMEngine,
    build_kv_transfer_config,
    empty_cuda_cache,
    force_vllm_inprocess_mode,
)
from inferscale.v1.types import SearchHit

from .composer import SampleComposer
from .composer_stats import precompute_stats
from .config import MemoryRunConfig
from .context import memory_context_metrics
from .data import ConversationSample, QuestionAnswer
from .prompting import MemoryQuery, build_memory_prompt_token_ids, build_memory_query
from .sample_cache import PrecomputedSamples
from .sample_precompute import SamplePrecompute
from .types import ComposedSampleMemory

COMPOSED_MEMORY_STORE_BACKEND = "gpu"


class KVAnswerClient:
    def __init__(self, config: MemoryRunConfig) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self.namespace = f"{config.run_id}-{uuid.uuid4().hex}"
        self.active_user_id = f"{self.namespace}-active"
        _create_composed_memory_store(self.namespace)
        self._engine = VLLMEngine(config.model, config.kv_dtype, config.engine_config())
        self._tokenizer: Any | None = None
        self._samples = PrecomputedSamples()
        self._precompute = SamplePrecompute(config)

    def precompute_sample_cache(
        self, sample: ConversationSample, facts: Sequence[SearchHit]
    ) -> dict[str, Any]:
        force_vllm_inprocess_mode()
        if self._engine.started:
            raise RuntimeError("KV precompute must finish for every sample before vLLM starts.")
        self._samples.release(id(sample))
        logger.info(
            "Precomputing KV cache sample_id={} facts={} context_window={}",
            sample.sample_id,
            len(facts),
            self.config.context_window,
        )
        started = time.perf_counter()
        composer, cache_hit = self._precompute.precompute(sample, facts)
        metrics = precompute_stats(composer)
        metrics.update(
            {
                "kv_precompute_time_ms": (time.perf_counter() - started) * 1000,
                "kv_chunk_cache_residency_is_gpu": int(self.config.kv_store_backend == "gpu"),
                "kv_chunk_cache_hit": int(cache_hit),
            }
        )
        self._samples.put(id(sample), composer, metrics)
        logger.info(
            "Precomputed KV cache sample_id={} chunks={} tokens={} layers={} gpu_mb={:.1f} "
            "devices={}",
            sample.sample_id,
            metrics.get("kv_precomputed_chunks", 0),
            metrics.get("kv_precomputed_tokens", 0),
            metrics.get("kv_precomputed_layers", 0),
            metrics.get("kv_precomputed_gpu_mb", 0.0),
            metrics.get("kv_precomputed_devices", ""),
        )
        return dict(metrics)

    def start_llm(self) -> None:
        force_vllm_inprocess_mode()
        if self._engine.started:
            return
        if not self._samples:
            raise RuntimeError("At least one sample must be precomputed before starting vLLM.")
        self._precompute.release_model()
        try:
            self._engine.start(
                kv_transfer_config=build_kv_transfer_config(
                    connector_module=self.config.kv_connector_module,
                    namespace=self.namespace,
                    default_user_id=self.active_user_id,
                    store_backend=COMPOSED_MEMORY_STORE_BACKEND,
                )
            )
            self._tokenizer = self._engine.tokenizer
        except Exception:
            self.close()
            raise

    def prepare_sample(self, sample: ConversationSample) -> None:
        if self._samples.active_key not in (None, id(sample)):
            self.close_sample()
        self._samples.activate(id(sample), sample.sample_id)
        logger.info("Prepared KV sample_id={}", sample.sample_id)

    def answer_with_retrieved_memory(
        self,
        *,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        max_tokens: int,
        temperature: float,
        top_p: float,
        ttft_started_at: float | None = None,
        query_started_at: float | None = None,
    ) -> ChatResult:
        composer = self._active_composer(sample)
        request_started = ttft_started_at if ttft_started_at is not None else time.perf_counter()
        query = build_memory_query(
            self._tokenizer,
            qa,
            block_size=self.config.kv_block_size,
            max_position=self.config.kv_max_position,
            max_model_len=self.config.kv_max_model_len,
            max_answer_tokens=max_tokens,
        )
        composed, transfer_metrics = self._compose(composer, hits, query)
        verify_ms = self._verify_against_prompt(composed, sample, hits, query)
        store_write_ms = self._register(composed)
        try:
            prompt = build_prompt_tokens(list(composed.token_ids), query.query_tokens)
            output = self._engine.generate(
                prompt.prompt_token_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            metrics: dict[str, Any] = {
                **self._samples.active_metrics,
                **_composed_memory_metrics(composed, prompt, self.config),
                "kv_verify_time_ms": verify_ms,
                "kv_store_write_time_ms": store_write_ms,
                **generation_latency_metrics(
                    output,
                    request_started=request_started,
                    query_started_at=query_started_at,
                    engine_ttft_key="kv_engine_time_to_first_token_ms",
                    excluded_ms=verify_ms + store_write_ms,
                ),
                "kv_store_gpu_mb": namespace_stats(self.namespace).get("total_gpu_mb", 0.0),
                "kv_store_host_mb": _host_store_mb(composer),
                **transfer_metrics,
            }
            return ChatResult(content=output.text, ttft_ms=output.ttft_ms, metrics=metrics)
        finally:
            remove_user_memory(self.namespace, self.active_user_id)

    def close_sample(self) -> None:
        self._samples.release_active()
        clear_namespace(self.namespace)
        empty_cuda_cache()

    def close(self) -> None:
        self._samples.release_all()
        clear_namespace(self.namespace)
        self._precompute.close()
        self._engine.close()
        self._tokenizer = None
        drop_namespace(self.namespace)
        empty_cuda_cache(collect_ipc=True)

    def _active_composer(self, sample: ConversationSample) -> SampleComposer:
        if not self._engine.started or self._tokenizer is None:
            raise RuntimeError("KVAnswerClient.start_llm() must be called before answering.")
        composer = self._samples.active_composer
        if composer is None or self._samples.active_key != id(sample):
            raise RuntimeError(
                f"KV sample_id={sample.sample_id} is not the prepared sample; "
                "call prepare_sample() first."
            )
        return composer

    def _compose(
        self, composer: SampleComposer, hits: list[SearchHit], query: MemoryQuery
    ) -> tuple[ComposedSampleMemory, dict[str, float]]:
        store = composer.chunk_store
        if store is None:
            raise RuntimeError("The precomputed sample has no chunk store.")
        totals_before = store.transfer_totals()
        composed = composer.compose(hits, memory_token_budget=query.memory_token_budget)
        return composed, _transfer_metrics(totals_before, store.transfer_totals())

    def _verify_against_prompt(
        self,
        composed: ComposedSampleMemory,
        sample: ConversationSample,
        hits: list[SearchHit],
        query: MemoryQuery,
    ) -> float:
        started = time.perf_counter()
        live_memory = build_memory_prompt_token_ids(
            self._tokenizer,
            sample,
            hits,
            context_window=self.config.context_window,
            memory_token_budget=query.memory_token_budget,
            memory_scaffold=query.scaffold,
        )
        require_identical_token_ids(composed.token_ids, live_memory.token_ids)
        return (time.perf_counter() - started) * 1000

    def _register(self, composed: ComposedSampleMemory) -> float:
        started = time.perf_counter()
        register_user_memory(
            self.namespace,
            user_id=self.active_user_id,
            kv_by_layer=composed.kv_by_layer,
            num_tokens=composed.num_tokens,
            token_ids=composed.token_ids,
        )
        return (time.perf_counter() - started) * 1000


def _create_composed_memory_store(namespace: str) -> None:
    get_gpu_memory_store(namespace, backend=COMPOSED_MEMORY_STORE_BACKEND)


def _composed_memory_metrics(
    composed: ComposedSampleMemory, prompt: PromptTokens, config: MemoryRunConfig
) -> dict[str, Any]:
    return {
        "kv_memory_tokens": composed.num_tokens,
        "kv_compose_time_ms": composed.compose_time_ms,
        "kv_context_window": composed.fact_plan.context_window,
        "kv_query_tokens": len(prompt.query_token_ids),
        "kv_query_bos_stripped": int(prompt.stripped_query_bos),
        "kv_selected_fact_ids": list(composed.fact_plan.injected_fact_ids),
        "kv_block_size": config.kv_block_size,
        "kv_prefix_caching": int(config.kv_enable_prefix_caching),
        "kv_loaded_memory_tokens": composed.loaded_memory_tokens,
        "kv_recomputed_memory_tail_tokens": composed.recomputed_memory_tail_tokens,
        "kv_fact_tokens_end": composed.fact_tokens_end,
        **memory_context_metrics(composed.fact_plan),
    }


def _transfer_metrics(
    totals_before: dict[str, float], totals_after: dict[str, float]
) -> dict[str, float]:
    h2d_ms = float(totals_after.get("total_h2d_latency_ms", 0.0)) - float(
        totals_before.get("total_h2d_latency_ms", 0.0)
    )
    stall_ms = float(totals_after.get("total_staging_stall_ms", 0.0)) - float(
        totals_before.get("total_staging_stall_ms", 0.0)
    )
    transferred_bytes = int(totals_after.get("total_bytes_transferred", 0)) - int(
        totals_before.get("total_bytes_transferred", 0)
    )
    overlap_ratio = max(0.0, min(1.0, 1.0 - stall_ms / h2d_ms)) if h2d_ms > 0 else 0.0
    return {
        "kv_h2d_latency_ms": h2d_ms,
        "kv_h2d_bytes": transferred_bytes,
        "kv_h2d_overlap_ratio": overlap_ratio,
        "kv_staging_stall_ms": stall_ms,
    }


def _host_store_mb(composer: SampleComposer) -> float:
    if composer.chunk_store is None:
        return 0.0
    return composer.chunk_store.get_stats().get("total_host_mb", 0.0)
