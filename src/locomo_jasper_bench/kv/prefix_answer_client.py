from __future__ import annotations

import gc
import logging
import time
from typing import Any

from ..clients import ChatResult
from ..config import BenchmarkConfig
from ..data import ConversationSample, QuestionAnswer
from ..vector_types import SearchHit
from .prompting import build_kv_equivalence_prompt_token_ids, build_memory_prompt_token_ids

logger = logging.getLogger(__name__)


class VLLMPrefixPromptAnswerClient:
    """In-process vLLM answer client for same-token KV-equivalence prompt injection."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_cls: Any | None = None
        self._active_samples: set[int] = set()

    def prepare_sample(self, sample: ConversationSample, hits_by_question: list[list[SearchHit]]) -> None:
        self.prepare_samples([(sample, hits_by_question)])

    def prepare_samples(
        self,
        samples: list[tuple[ConversationSample, list[list[SearchHit]]]],
    ) -> None:
        if not samples:
            raise RuntimeError("vllm-prefix mode cannot prepare an empty sample window.")
        if self.config.kv_sample_window < 1:
            raise RuntimeError("--kv-sample-window must be >= 1.")
        if len(samples) > self.config.kv_sample_window:
            raise RuntimeError(
                f"vllm-prefix sample window got {len(samples)} samples, "
                f"exceeding --kv-sample-window {self.config.kv_sample_window}."
            )

        self.close_sample()

        from vllm import LLM, SamplingParams

        try:
            logger.info("Preparing vLLM prefix prompt sample window size=%d", len(samples))
            self._sampling_cls = SamplingParams
            self._llm = LLM(
                model=self.config.model,
                dtype=self.config.kv_dtype,
                trust_remote_code=True,
                enable_prefix_caching=False,
                swap_space=0,
                cpu_offload_gb=0,
                gpu_memory_utilization=self.config.kv_gpu_memory_utilization,
                max_model_len=self.config.kv_max_model_len,
            )
            self._tokenizer = self._llm.get_tokenizer()
            self._active_samples = {id(sample) for sample, _ in samples}
        except Exception:
            self.close_sample()
            raise

    def answer_with_retrieved_memory(
        self,
        *,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> ChatResult:
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
            raise RuntimeError("VLLMPrefixPromptAnswerClient.prepare_sample() must be called before answering.")
        if id(sample) not in self._active_samples:
            raise RuntimeError(f"vllm-prefix sample_id={sample.sample_id} is not in the active sample window.")

        memory = build_memory_prompt_token_ids(self._tokenizer, sample, hits)
        prompt = build_kv_equivalence_prompt_token_ids(
            self._tokenizer,
            memory.token_ids,
            sample,
            qa,
        )

        ttft_ms = self._measure_one_token_ttft(
            prompt_token_ids=prompt.prompt_token_ids,
            temperature=temperature,
            top_p=top_p,
        )

        sampling = self._sampling_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self._llm.generate(
            [{"prompt_token_ids": prompt.prompt_token_ids}],
            sampling,
            use_tqdm=False,
        )
        return ChatResult(
            content=outputs[0].outputs[0].text.strip(),
            ttft_ms=ttft_ms,
        )

    def close_sample(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._tokenizer = None
        self._sampling_cls = None
        self._active_samples.clear()
        gc.collect()
        _empty_cuda_cache()

    def close(self) -> None:
        self.close_sample()

    def _measure_one_token_ttft(
        self,
        *,
        prompt_token_ids: list[int],
        temperature: float,
        top_p: float,
    ) -> float:
        sampling = self._sampling_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=1,
            min_tokens=1,
        )
        _synchronize_cuda()
        engine_started = time.perf_counter()
        self._llm.generate(
            [{"prompt_token_ids": prompt_token_ids}],
            sampling,
            use_tqdm=False,
        )
        _synchronize_cuda()
        finished = time.perf_counter()
        return (finished - engine_started) * 1000


def _synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
