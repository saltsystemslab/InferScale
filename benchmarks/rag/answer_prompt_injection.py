"""The prompt-injection baseline: the same chunk token ids rendered as prompt text."""

from __future__ import annotations

import time
from typing import Any

from benchmarks.common.answer_timing import generation_latency_metrics
from benchmarks.common.clients import ChatResult
from benchmarks.rag.data_types import RagQuery
from benchmarks.rag.engine_client import RagEngineClient
from inferscale.v1.kv.compose import reverse_ranked_ids
from inferscale.v1.kv.prompt import build_memory_token_ids, build_prompt_tokens
from inferscale.v1.serving.vllm import empty_cuda_cache
from inferscale.v1.types import SearchHit


class RagPromptInjectionAnswerClient(RagEngineClient):
    label = "prompt-injection"

    def answer(
        self,
        query: RagQuery,
        hits: list[SearchHit],
        *,
        query_started_at: float | None = None,
    ) -> ChatResult:
        scaffold = self._started_scaffold()
        request_started = time.perf_counter()
        ordered_chunk_ids = reverse_ranked_ids(hits)
        memory_token_ids = build_memory_token_ids(
            scaffold, [self._chunk_token_ids(chunk_id) for chunk_id in ordered_chunk_ids]
        )
        query_tokens = self._query_tokens(memory_token_ids, query)
        self._require_budget(memory_token_ids, query_tokens)
        prompt = build_prompt_tokens(memory_token_ids, query_tokens)
        output = self._generate(prompt)
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(prompt.memory_token_ids),
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_selected_chunk_ids": ordered_chunk_ids,
            "kv_block_size": self.config.kv_block_size,
            "retrieved_chunk_count": len(hits),
            **generation_latency_metrics(
                output,
                request_started=request_started,
                query_started_at=query_started_at,
                engine_ttft_key="prompt_injection_engine_time_to_first_token_ms",
            ),
        }
        return ChatResult(content=output.text, ttft_ms=output.ttft_ms, metrics=metrics)

    def close(self) -> None:
        super().close()
        empty_cuda_cache()
