"""Answer LoCoMo questions with retrieved Mem0 facts injected as prompt text.

The baseline sends exactly the token stream KV injection would compose, so
the two backends differ only in whether vLLM recomputes the memory prefix.
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from benchmarks.common.answer_timing import generation_latency_metrics
from benchmarks.common.clients import ChatResult
from inferscale.v1.kv.prompt import build_prompt_tokens
from inferscale.v1.serving.vllm import VLLMEngine, empty_cuda_cache, force_vllm_inprocess_mode
from inferscale.v1.types import SearchHit

from .config import MemoryRunConfig
from .context import memory_context_metrics
from .data import ConversationSample, QuestionAnswer
from .prompting import build_memory_prompt_token_ids, build_memory_query


class PromptInjectionAnswerClient:
    """In-process vLLM answer client for same-token prompt injection."""

    def __init__(self, config: MemoryRunConfig) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self._engine = VLLMEngine(config.model, config.kv_dtype, config.engine_config())
        self._tokenizer: Any | None = None
        self._active_sample_key: int | None = None

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
        self._active_sample_key = id(sample)
        logger.info("Prepared prompt-injection sample_id={}", sample.sample_id)

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
            raise RuntimeError(
                "PromptInjectionAnswerClient.start_llm() must be called before answering."
            )
        if self._active_sample_key != id(sample):
            raise RuntimeError(
                f"prompt-injection sample_id={sample.sample_id} is not the prepared sample; "
                "call prepare_sample() first."
            )
        request_started = ttft_started_at if ttft_started_at is not None else time.perf_counter()
        query = build_memory_query(
            self._tokenizer,
            qa,
            block_size=self.config.kv_block_size,
            max_position=self.config.kv_max_position,
            max_model_len=self.config.kv_max_model_len,
            max_answer_tokens=max_tokens,
        )
        memory = build_memory_prompt_token_ids(
            self._tokenizer,
            sample,
            hits,
            context_window=self.config.context_window,
            memory_token_budget=query.memory_token_budget,
            memory_scaffold=query.scaffold,
            render_context_turns=True,
        )
        prompt = build_prompt_tokens(memory.token_ids, query.query_tokens)
        output = self._engine.generate(
            prompt.prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(prompt.memory_token_ids),
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_selected_fact_ids": list(memory.fact_plan.injected_fact_ids),
            "kv_block_size": self.config.kv_block_size,
            "kv_context_window": memory.fact_plan.context_window,
            **memory_context_metrics(memory.fact_plan),
            **generation_latency_metrics(
                output,
                request_started=request_started,
                query_started_at=query_started_at,
                engine_ttft_key="prompt_injection_engine_time_to_first_token_ms",
            ),
        }
        return ChatResult(content=output.text, ttft_ms=output.ttft_ms, metrics=metrics)

    def close_sample(self) -> None:
        self._active_sample_key = None

    def close(self) -> None:
        self.close_sample()
        self._engine.close()
        self._tokenizer = None
        empty_cuda_cache()
