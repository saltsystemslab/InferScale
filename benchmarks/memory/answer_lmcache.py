from __future__ import annotations

import time
from typing import Any

from loguru import logger

from benchmarks.common.answer_timing import generation_latency_metrics
from benchmarks.common.clients import ChatResult
from benchmarks.common.lmcache.config import engine_config_for_lmcache, lmcache_kv_transfer_config
from benchmarks.common.lmcache.load_stats import reset_load_stats, snapshot_load_stats
from benchmarks.common.lmcache.runtime import (
    TAIL_ALLOWANCE_TOKENS,
    clear_cache,
    kv_bytes_per_token,
    lmcache_separator,
    register_model,
    require_cache_capacity,
    shutdown,
    warm_prompts,
    warmup_metrics,
)
from benchmarks.common.lmcache.segments import (
    blend_memory_layout,
    prompt_token_count,
    warm_up_prompts,
)
from inferscale.v1.kv.prompt import build_prompt_tokens
from inferscale.v1.serving.vllm import VLLMEngine, empty_cuda_cache, force_vllm_inprocess_mode
from inferscale.v1.types import SearchHit

from .config import MemoryRunConfig
from .context import memory_context_metrics, plan_prompt_memory
from .data import ConversationSample, QuestionAnswer
from .prompting import build_memory_query, extract_memory_scaffold_token_ids, select_memory_facts

WARM_BATCH_SIZE = 256


class LMCacheBlendAnswerClient:
    def __init__(self, config: MemoryRunConfig) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self._lmcache_settings = config.lmcache_settings()
        self._engine = VLLMEngine(
            config.model, config.kv_dtype, engine_config_for_lmcache(config.engine_config())
        )
        self._tokenizer: Any | None = None
        self._separator: list[int] | None = None
        self._kv_bytes_per_token = 0
        self._active_sample_key: int | None = None

    def start_llm(self) -> None:
        if self._engine.started:
            return
        force_vllm_inprocess_mode()
        try:
            self._engine.start(
                kv_transfer_config=lmcache_kv_transfer_config(self._lmcache_settings)
            )
            register_model(self._engine.llm)
            self._tokenizer = self._engine.tokenizer
            self._separator = lmcache_separator()
            self._kv_bytes_per_token = kv_bytes_per_token()
        except Exception:
            self.close()
            raise
        logger.info(
            "Started lmcache-blend engine model={} separator_tokens={} kv_bytes_per_token={}",
            self.config.model,
            len(self._separator),
            self._kv_bytes_per_token,
        )

    def prepare_sample(self, sample: ConversationSample) -> None:
        self._active_sample_key = id(sample)
        logger.info("Prepared lmcache-blend sample_id={}", sample.sample_id)

    def warm_sample_memory(
        self, sample: ConversationSample, facts: list[SearchHit]
    ) -> dict[str, Any]:
        tokenizer, separator = self._require_started()
        clear_cache()
        scaffold = extract_memory_scaffold_token_ids(
            tokenizer, block_size=self.config.kv_block_size
        )
        catalog_facts = select_memory_facts(tokenizer, facts)
        single_segment_prompts = warm_up_prompts(
            scaffold.header_token_ids, catalog_facts.ordered_token_ids(), separator
        )
        require_cache_capacity(
            warm_tokens=prompt_token_count(single_segment_prompts),
            tail_tokens=TAIL_ALLOWANCE_TOKENS * len(sample.qa),
            bytes_per_token=self._kv_bytes_per_token,
            max_local_cpu_size_gb=self._lmcache_settings.max_local_cpu_size_gb,
            what=f"sample_id={sample.sample_id} fact catalog",
        )
        warmup_stats = warm_prompts(
            self._engine.llm, single_segment_prompts, batch_size=WARM_BATCH_SIZE
        )
        logger.info(
            "Warmed LMCache for sample_id={} prompts={} tokens={} in {:.0f} ms",
            sample.sample_id,
            warmup_stats.prompt_count,
            warmup_stats.token_count,
            warmup_stats.time_ms,
        )
        return warmup_metrics(warmup_stats)

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
        tokenizer, separator = self._require_started()
        if self._active_sample_key != id(sample):
            raise RuntimeError(
                f"lmcache-blend sample_id={sample.sample_id} is not the prepared sample; "
                "call prepare_sample() first."
            )
        request_started = ttft_started_at if ttft_started_at is not None else time.perf_counter()
        query = build_memory_query(
            tokenizer,
            qa,
            block_size=self.config.kv_block_size,
            max_position=self.config.kv_max_position,
            max_model_len=self.config.kv_max_model_len,
            max_answer_tokens=max_tokens,
        )
        facts = select_memory_facts(tokenizer, hits)
        layout = blend_memory_layout(query.scaffold, facts.ordered_token_ids(), separator)
        fact_plan = plan_prompt_memory(
            facts.selected,
            retrieved_facts=facts.retrieved,
            context_window=0,
            memory_token_budget=query.memory_token_budget,
            scaffold_token_count=layout.scaffold_tokens,
            fact_token_ids=facts.token_ids,
        )
        if len(layout.token_ids) != fact_plan.memory_tokens:
            raise AssertionError("Planned memory token count does not match the blend layout.")
        prompt = build_prompt_tokens(layout.token_ids, query.query_tokens)
        reset_load_stats()
        output = self._engine.generate(
            prompt.prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        load_stats = snapshot_load_stats()
        if load_stats["tokens_matched"] < layout.expected_matched_tokens:
            logger.warning(
                "LMCache matched {} of {} expected memory tokens for sample_id={} question_id={}",
                load_stats["tokens_matched"],
                layout.expected_matched_tokens,
                sample.sample_id,
                qa.question_id,
            )
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(prompt.memory_token_ids),
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_selected_fact_ids": list(fact_plan.injected_fact_ids),
            "kv_block_size": self.config.kv_block_size,
            "kv_context_window": fact_plan.context_window,
            "kv_prefix_caching": 0,
            "lmcache_separator_tokens": len(separator),
            "lmcache_segment_count": layout.segment_count,
            "lmcache_expected_matched_tokens": layout.expected_matched_tokens,
            "lmcache_matched_tokens": load_stats["tokens_matched"],
            "lmcache_load_time_ms": load_stats["load_time_ms"],
            **memory_context_metrics(fact_plan),
            **generation_latency_metrics(
                output,
                request_started=request_started,
                query_started_at=query_started_at,
                engine_ttft_key="lmcache_engine_time_to_first_token_ms",
            ),
        }
        return ChatResult(content=output.text, ttft_ms=output.ttft_ms, metrics=metrics)

    def close_sample(self) -> None:
        self._active_sample_key = None

    def close(self) -> None:
        self.close_sample()
        shutdown()
        self._engine.close()
        self._tokenizer = None
        self._separator = None
        empty_cuda_cache()

    def _require_started(self) -> tuple[Any, list[int]]:
        if not self._engine.started or self._tokenizer is None or self._separator is None:
            raise RuntimeError(
                "LMCacheBlendAnswerClient.start_llm() must be called before answering."
            )
        return self._tokenizer, self._separator
