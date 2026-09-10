from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from loguru import logger

from inferscale.v1.kv.prompt import build_prompt_tokens
from inferscale.v1.serving.vllm import VLLMEngine, empty_cuda_cache, force_vllm_inprocess_mode
from inferscale.v1.types import SearchHit

from benchmarks.common.clients import ChatResult

from .config import RagBenchConfig
from .data_types import RagChunk, RagPromptProfile, RagQuery
from .prompting import (
    build_rag_memory_token_ids,
    build_rag_query_tokens,
    calculate_rag_memory_budget,
    extract_rag_scaffold_token_ids,
    require_memory_within_budget,
    reverse_ranked_chunk_ids,
)


class RagPrefixAnswerClient:
    """Text-prompt baseline: identical chunk token ids stuffed as a plain prefix."""

    def __init__(
        self,
        config: RagBenchConfig,
        *,
        chunks_by_id: Mapping[str, RagChunk],
        prompt_profile: RagPromptProfile,
    ) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self._chunks_by_id = chunks_by_id
        self._prompt_profile = prompt_profile
        self._engine = VLLMEngine(config.model, config.kv_dtype, config.engine_config())
        self._tokenizer: Any | None = None
        self._scaffold: Any | None = None

    def start_llm(self) -> None:
        if self._engine.started:
            return
        force_vllm_inprocess_mode()
        try:
            self._engine.start()
            self._tokenizer = self._engine.tokenizer
            self._scaffold = extract_rag_scaffold_token_ids(
                self._tokenizer,
                system_prompt=self._prompt_profile.system_prompt,
                block_size=self.config.kv_block_size,
            )
        except Exception:
            self.close()
            raise
        logger.info("Started prompt-injection answer engine model={}", self.config.model)

    def answer(
        self,
        query: RagQuery,
        hits: list[SearchHit],
        *,
        query_started_at: float | None = None,
    ) -> ChatResult:
        if not self._engine.started or self._tokenizer is None:
            raise RuntimeError("RagPrefixAnswerClient.start_llm() must be called before answering.")
        request_started = time.perf_counter()
        ordered_chunk_ids = reverse_ranked_chunk_ids(hits)
        memory_token_ids = build_rag_memory_token_ids(
            self._scaffold,
            [self._chunk_token_ids(chunk_id) for chunk_id in ordered_chunk_ids],
        )
        query_tokens = build_rag_query_tokens(
            self._tokenizer,
            memory_token_ids,
            query,
            answer_instruction=self._prompt_profile.answer_instruction,
        )
        memory_token_budget = calculate_rag_memory_budget(
            query_token_count=len(query_tokens.token_ids),
            max_position=self.config.kv_max_position,
            max_model_len=self.config.kv_max_model_len,
            max_answer_tokens=self.config.max_answer_tokens,
        )
        require_memory_within_budget(
            len(memory_token_ids),
            memory_token_budget,
            top_k=self.config.top_k,
            chunk_size=self.config.chunk_size,
        )
        prompt = build_prompt_tokens(memory_token_ids, query_tokens)
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(prompt.memory_token_ids),
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_selected_chunk_ids": ordered_chunk_ids,
            "kv_block_size": self.config.kv_block_size,
            "retrieved_chunk_count": len(hits),
        }

        output = self._engine.generate(
            prompt.prompt_token_ids,
            max_tokens=self.config.max_answer_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
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
        metrics["answer_time_to_first_token_ms"] = (
            max(0.0, (generate_started - request_started) * 1000) + ttft_ms
        )
        if query_started_at is not None:
            metrics["query_to_first_token_ms"] = (
                max(0.0, (generate_started - query_started_at) * 1000) + ttft_ms
            )
        return ChatResult(
            content=output.text,
            ttft_ms=ttft_ms,
            metrics=metrics,
        )

    def close(self) -> None:
        self._engine.close()
        self._tokenizer = None
        self._scaffold = None
        empty_cuda_cache()

    def _chunk_token_ids(self, chunk_id: str) -> list[int]:
        chunk = self._chunks_by_id.get(chunk_id)
        if chunk is None:
            raise RuntimeError(f"Retrieved chunk id {chunk_id} is not part of the corpus.")
        return chunk.token_ids
