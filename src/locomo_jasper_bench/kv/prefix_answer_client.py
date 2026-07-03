from __future__ import annotations

import logging
import time
from typing import Any

from ..clients import ChatResult
from ..config import BenchmarkConfig
from ..data import ConversationSample, QuestionAnswer
from ..vector_types import SearchHit
from .prompting import build_kv_equivalence_prompt_token_ids, build_memory_prompt_token_ids
from .vllm_metrics import request_timing_from_output
from .vllm_runtime import (
    common_vllm_kwargs,
    empty_cuda_cache,
    sanitize_repo_vllm_env_for_import,
)

logger = logging.getLogger(__name__)


class VLLMPrefixPromptAnswerClient:
    """In-process vLLM answer client for same-token KV-equivalence prompt injection."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_cls: Any | None = None
        self._active_sample_id: str | None = None

    def start_llm(self) -> None:
        if self._llm is not None:
            return

        sanitize_repo_vllm_env_for_import()
        from vllm import LLM, SamplingParams

        try:
            self._sampling_cls = SamplingParams
            self._llm = LLM(**common_vllm_kwargs(self.config))
            self._tokenizer = self._llm.get_tokenizer()
        except Exception:
            self.close()
            raise

    def prepare_sample(self, sample: ConversationSample) -> None:
        self.close_sample()
        logger.info("Preparing vLLM prefix prompt sample_id=%s", sample.sample_id)
        self._active_sample_id = sample.sample_id

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
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
            raise RuntimeError("VLLMPrefixPromptAnswerClient.prepare_sample() must be called before answering.")
        if self._active_sample_id != sample.sample_id:
            raise RuntimeError(f"vllm-prefix sample_id={sample.sample_id} is not the active prepared sample.")

        request_started = ttft_started_at if ttft_started_at is not None else time.perf_counter()
        memory = build_memory_prompt_token_ids(
            self._tokenizer,
            sample,
            hits,
            memory_order=self.config.memory_order,
        )
        prompt = build_kv_equivalence_prompt_token_ids(
            self._tokenizer,
            memory.token_ids,
            sample,
            qa,
        )
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(prompt.memory_token_ids),
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_retrieval_session_ids": memory.retrieval_session_ids,
            "kv_selected_session_ids": memory.selected_session_ids,
            "kv_memory_order": memory.memory_order,
        }

        sampling = self._sampling_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
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
        timing = request_timing_from_output(outputs[0])
        ttft_ms = timing.time_to_first_token_ms
        if ttft_ms is not None:
            metrics["prefix_engine_time_to_first_token_ms"] = ttft_ms
            metrics["answer_time_to_first_token_ms"] = max(0.0, (generate_started - request_started) * 1000) + ttft_ms
            if query_started_at is not None:
                metrics["query_to_first_token_ms"] = max(0.0, (generate_started - query_started_at) * 1000) + ttft_ms
        return ChatResult(
            content=outputs[0].outputs[0].text.strip(),
            ttft_ms=ttft_ms,
            metrics=metrics,
        )

    def close_sample(self) -> None:
        self._active_sample_id = None

    def close(self) -> None:
        self.close_sample()
        if self._llm is not None:
            del self._llm
        self._llm = None
        self._tokenizer = None
        self._sampling_cls = None
        empty_cuda_cache()
