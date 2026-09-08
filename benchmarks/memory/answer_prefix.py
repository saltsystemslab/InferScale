from __future__ import annotations

import logging
import time
from typing import Any

from inferscale.v1.kv.prompt import build_prompt_tokens
from inferscale.v1.serving.vllm import VLLMEngine, empty_cuda_cache, force_vllm_inprocess_mode
from inferscale.v1.types import SearchHit

from benchmarks.common.clients import ChatResult

from .config import MemoryRunConfig
from .context import memory_context_metrics
from .data import ConversationSample, QuestionAnswer
from .prompting import (
    build_kv_query_tokens_for_memory,
    build_memory_prompt_token_ids,
    calculate_memory_token_budget,
    extract_memory_scaffold_token_ids,
)

logger = logging.getLogger(__name__)


class PrefixAnswerClient:
    """In-process vLLM answer client for same-token KV-equivalence prompt injection."""

    def __init__(self, config: MemoryRunConfig) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self._engine = VLLMEngine(config.model, config.kv_dtype, config.engine_config())
        self._tokenizer: Any | None = None
        self._active_sample_id: int | None = None

    def start_llm(self) -> None:
        if self._engine.started:
            return
        force_vllm_inprocess_mode()
        try:
            self._engine.start()
            self._tokenizer = self._engine.tokenizer
        except Exception:
            self.close()
            raise

    def prepare_sample(self, sample: ConversationSample) -> None:
        self.close_sample()
        logger.info("Preparing prompt-injection prompt sample_id=%s", sample.sample_id)
        self._active_sample_id = id(sample)

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
        if not self._engine.started or self._tokenizer is None:
            raise RuntimeError("PrefixAnswerClient.prepare_sample() must be called before answering.")
        if self._active_sample_id != id(sample):
            raise RuntimeError(f"prompt-injection sample_id={sample.sample_id} is not the active prepared sample.")

        request_started = ttft_started_at if ttft_started_at is not None else time.perf_counter()
        scaffold = extract_memory_scaffold_token_ids(
            self._tokenizer,
            block_size=self.config.kv_block_size,
        )
        query_tokens = build_kv_query_tokens_for_memory(
            self._tokenizer,
            scaffold.header_token_ids,
            sample,
            qa,
        )
        memory_token_budget = calculate_memory_token_budget(
            self._tokenizer,
            sample,
            qa,
            memory_prefix_token_ids=scaffold.header_token_ids,
            max_position=self.config.kv_max_position,
            max_model_len=self.config.kv_max_model_len,
            max_answer_tokens=max_tokens,
            query_tokens=query_tokens,
        )
        memory = build_memory_prompt_token_ids(
            self._tokenizer,
            sample,
            hits,
            context_window=self.config.context_window,
            memory_token_budget=memory_token_budget,
            memory_scaffold=scaffold,
            render_context_turns=True,
        )
        prompt = build_prompt_tokens(
            memory.token_ids,
            query_tokens,
        )
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(prompt.memory_token_ids),
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_selected_fact_ids": memory.selected_fact_ids,
            "kv_block_size": self.config.kv_block_size,
            "kv_context_window": memory.fact_plan.context_window,
            **memory_context_metrics(memory.fact_plan),
        }

        output = self._engine.generate(
            prompt.prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        generate_started = output.started_at
        finished = output.finished_at
        metrics.update(
            {
                "answer_generate_time_ms": (finished - generate_started) * 1000,
                "answer_total_time_ms": max(0.0, (finished - request_started) * 1000),
            }
        )
        if query_started_at is not None:
            metrics["query_to_answer_ms"] = max(0.0, (finished - query_started_at) * 1000)
        ttft_ms = output.ttft_ms
        metrics["prefix_engine_time_to_first_token_ms"] = ttft_ms
        metrics["answer_time_to_first_token_ms"] = max(0.0, (generate_started - request_started) * 1000) + ttft_ms
        if query_started_at is not None:
            metrics["query_to_first_token_ms"] = max(0.0, (generate_started - query_started_at) * 1000) + ttft_ms
        return ChatResult(
            content=output.text,
            ttft_ms=ttft_ms,
            metrics=metrics,
        )

    def close_sample(self) -> None:
        self._active_sample_id = None

    def close(self) -> None:
        self.close_sample()
        self._engine.close()
        self._tokenizer = None
        empty_cuda_cache()
