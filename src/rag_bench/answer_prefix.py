from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from loguru import logger

from locomo_jasper_bench.clients import ChatResult
from locomo_jasper_bench.kv.prompting import build_kv_equivalence_prompt_from_query_tokens
from locomo_jasper_bench.kv.vllm_metrics import require_engine_ttft_ms
from locomo_jasper_bench.kv.vllm_runtime import (
    common_vllm_kwargs,
    empty_cuda_cache,
    force_vllm_inprocess_mode,
)
from locomo_jasper_bench.vector_types import SearchHit

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
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_cls: Any | None = None
        self._scaffold: Any | None = None

    def start_llm(self) -> None:
        if self._llm is not None:
            return
        force_vllm_inprocess_mode()
        from vllm import LLM, SamplingParams

        try:
            self._sampling_cls = SamplingParams
            self._llm = LLM(**common_vllm_kwargs(self.config))
            self._tokenizer = self._llm.get_tokenizer()
            self._scaffold = extract_rag_scaffold_token_ids(
                self._tokenizer,
                system_prompt=self._prompt_profile.system_prompt,
                block_size=self.config.kv_block_size,
            )
        except Exception:
            self.close()
            raise
        logger.info("Started vLLM prefix answer engine model={}", self.config.model)

    def answer(
        self,
        query: RagQuery,
        hits: list[SearchHit],
        *,
        query_started_at: float | None = None,
    ) -> ChatResult:
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
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
        prompt = build_kv_equivalence_prompt_from_query_tokens(memory_token_ids, query_tokens)
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(prompt.memory_token_ids),
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_selected_chunk_ids": ordered_chunk_ids,
            "kv_block_size": self.config.kv_block_size,
            "retrieved_chunk_count": len(hits),
        }

        sampling = self._sampling_cls(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_answer_tokens,
        )
        generate_started = time.perf_counter()
        outputs = self._llm.generate(
            [{"prompt_token_ids": prompt.prompt_token_ids}],
            sampling,
            use_tqdm=False,
        )
        finished = time.perf_counter()
        metrics.update(
            {
                "answer_generate_time_ms": (finished - generate_started) * 1000,
                "answer_total_time_ms": max(0.0, (finished - request_started) * 1000),
            }
        )
        if query_started_at is not None:
            metrics["query_to_answer_ms"] = max(0.0, (finished - query_started_at) * 1000)
        ttft_ms = require_engine_ttft_ms(outputs[0])
        metrics["prefix_engine_time_to_first_token_ms"] = ttft_ms
        metrics["answer_time_to_first_token_ms"] = (
            max(0.0, (generate_started - request_started) * 1000) + ttft_ms
        )
        if query_started_at is not None:
            metrics["query_to_first_token_ms"] = (
                max(0.0, (generate_started - query_started_at) * 1000) + ttft_ms
            )
        return ChatResult(
            content=outputs[0].outputs[0].text.strip(),
            ttft_ms=ttft_ms,
            metrics=metrics,
        )

    def close(self) -> None:
        if self._llm is not None:
            del self._llm
        self._llm = None
        self._tokenizer = None
        self._sampling_cls = None
        self._scaffold = None
        empty_cuda_cache()

    def _chunk_token_ids(self, chunk_id: str) -> list[int]:
        chunk = self._chunks_by_id.get(chunk_id)
        if chunk is None:
            raise RuntimeError(f"Retrieved chunk id {chunk_id} is not part of the corpus.")
        return chunk.token_ids
