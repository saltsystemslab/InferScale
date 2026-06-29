from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from ..clients import ChatResult
from ..config import BenchmarkConfig
from ..data import ConversationSample, QuestionAnswer
from ..vector_types import SearchHit
from .chunked_rope import ChunkedRopeSampleComposer
from .gpu_registry import (
    clear_namespace,
    drop_namespace,
    namespace_stats,
    register_user_memory,
    remove_user_memory,
)
from .prompting import build_kv_equivalence_prompt_token_ids
from .sample_cache import GpuSampleCacheStore
from .ai_memory_code import require_ai_memory_code
from .vllm_metrics import request_timing_from_output
from .vllm_runtime import (
    build_strict_gpu_kv_transfer_config,
    common_vllm_kwargs,
    empty_cuda_cache,
    force_vllm_inprocess_mode,
)

logger = logging.getLogger(__name__)


class VLLMChunkedKVAnswerClient:
    """In-process vLLM answer client using strict GPU chunked-RoPE KV injection."""

    def __init__(self, config: BenchmarkConfig) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self.namespace = f"{config.run_id}-{uuid.uuid4().hex}"
        self.active_user_id = f"{self.namespace}-active"
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_cls: Any | None = None
        self._sample_caches = GpuSampleCacheStore()

    def precompute_sample_cache(self, sample: ConversationSample) -> dict[str, Any]:
        force_vllm_inprocess_mode()
        if self._llm is not None:
            raise RuntimeError(
                "GPU-resident KV precompute must run before vLLM is started. "
                "Precompute all needed samples before starting the single vLLM instance."
            )
        require_ai_memory_code()
        sample_key = id(sample)
        if self._sample_caches.active_sample_key == sample_key:
            self.close_sample()
        else:
            self._sample_caches.release(sample_key)
        logger.info(
            "Precomputing GPU-resident KV cache sample_id=%s turns=%d context_window=%d",
            sample.sample_id,
            len(sample.turns),
            self.config.context_window,
        )
        started = time.perf_counter()
        composer: ChunkedRopeSampleComposer | None = None
        try:
            composer = ChunkedRopeSampleComposer(
                model=self.config.model,
                dtype=self.config.kv_dtype,
                device=self.config.kv_device,
                max_position=self.config.kv_max_position,
                context_window=self.config.context_window,
            )
            composer.encode_sample(sample)
            composer.release_encoder()
        except RuntimeError as exc:
            if composer is not None:
                composer.close()
            raise RuntimeError(
                f"GPU-resident KV precompute failed for sample_id={sample.sample_id}. "
                "No CPU or disk KV fallback was used."
            ) from exc
        except BaseException:
            if composer is not None:
                composer.close()
            raise

        sample_metrics = composer.cache_stats()
        sample_metrics.update(
            {
                "kv_precompute_time_ms": (time.perf_counter() - started) * 1000,
                "kv_chunk_cache_residency_is_gpu": 1,
            }
        )
        self._sample_caches.put(sample_key, composer, sample_metrics)
        logger.info(
            "Precomputed GPU-resident KV cache sample_id=%s chunks=%s tokens=%s layers=%s gpu_mb=%.1f devices=%s",
            sample.sample_id,
            sample_metrics.get("kv_precomputed_chunks", 0),
            sample_metrics.get("kv_precomputed_tokens", 0),
            sample_metrics.get("kv_precomputed_layers", 0),
            sample_metrics.get("kv_precomputed_gpu_mb", 0.0),
            sample_metrics.get("kv_precomputed_devices", ""),
        )
        return dict(sample_metrics)

    def start_llm(self) -> None:
        force_vllm_inprocess_mode()
        if self._llm is not None:
            return
        if not self._sample_caches:
            raise RuntimeError("At least one GPU-resident KV sample cache must be precomputed before starting vLLM.")

        from vllm import LLM, SamplingParams

        try:
            self._sampling_cls = SamplingParams
            self._llm = LLM(
                **common_vllm_kwargs(self.config),
                kv_transfer_config=build_strict_gpu_kv_transfer_config(
                    connector_module=self.config.kv_connector_module,
                    namespace=self.namespace,
                    default_user_id=self.active_user_id,
                ),
            )
            self._tokenizer = self._llm.get_tokenizer()
        except Exception:
            self.close()
            raise

    def prepare_sample(self, sample: ConversationSample) -> None:
        sample_key = id(sample)
        if self._sample_caches.active_sample_key is not None and self._sample_caches.active_sample_key != sample_key:
            self.close_sample()
        self._sample_caches.prepare(sample_key, sample.sample_id)
        logger.info("Preparing GPU-resident KV sample_id=%s", sample.sample_id)

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
            raise RuntimeError("VLLMChunkedKVAnswerClient.prepare_sample() must be called before answering.")
        composer = self._sample_caches.active_composer
        if composer is None:
            raise RuntimeError(f"Strict GPU KV cache sample_id={sample.sample_id} was not prepared.")
        if self._sample_caches.active_sample_key != id(sample):
            raise RuntimeError(f"Strict GPU KV sample_id={sample.sample_id} is not the active prepared sample.")

        request_started = ttft_started_at if ttft_started_at is not None else time.perf_counter()
        composed = composer.compose(hits)
        user_id = self.active_user_id
        register_user_memory(
            self.namespace,
            user_id=user_id,
            kv_by_layer=composed.kv_by_layer,
            num_tokens=composed.num_tokens,
            token_ids=composed.token_ids,
            memory_text="strict-gpu chunked-rope top-k",
        )
        try:
            prompt = build_kv_equivalence_prompt_token_ids(
                self._tokenizer,
                list(composed.token_ids),
                sample,
                qa,
            )
            metrics: dict[str, Any] = {
                **self._sample_caches.active_metrics,
                "kv_memory_tokens": composed.num_tokens,
                "kv_compose_time_ms": composed.compose_time_ms,
                "kv_context_window": composed.context_window,
                "kv_context_prefix_tokens_total": composed.context_prefix_tokens_total,
                "kv_context_prefix_tokens_max": composed.context_prefix_tokens_max,
                "kv_context_prefix_truncated_tokens": composed.context_prefix_truncated_tokens,
                "kv_query_tokens": len(prompt.query_token_ids),
                "kv_query_bos_stripped": int(prompt.stripped_query_bos),
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
            generate_ms = (finished - generate_started) * 1000
            total_ms = max(0.0, (finished - request_started) * 1000)
            if query_started_at is not None:
                metrics["query_to_answer_ms"] = max(0.0, (finished - query_started_at) * 1000)
            timing = request_timing_from_output(outputs[0])
            ttft_ms = timing.time_to_first_token_ms
            if ttft_ms is not None:
                metrics["kv_engine_time_to_first_token_ms"] = ttft_ms
                metrics["answer_time_to_first_token_ms"] = max(0.0, (generate_started - request_started) * 1000) + ttft_ms
                if query_started_at is not None:
                    metrics["query_to_first_token_ms"] = max(0.0, (generate_started - query_started_at) * 1000) + ttft_ms
            text = outputs[0].outputs[0].text.strip()
            stats = namespace_stats(self.namespace)
            metrics.update(
                {
                    "answer_generate_time_ms": generate_ms,
                    "answer_total_time_ms": total_ms,
                    "kv_store_gpu_mb": stats.get("total_gpu_mb", 0.0),
                    "kv_selected_turn_ids": composed.selected_turn_ids,
                }
            )
            return ChatResult(
                content=text,
                ttft_ms=ttft_ms,
                metrics=metrics,
            )
        finally:
            remove_user_memory(self.namespace, user_id)

    def close_sample(self) -> None:
        self._sample_caches.release_active()
        clear_namespace(self.namespace)
        empty_cuda_cache()

    def close(self) -> None:
        self._sample_caches.release_all()
        clear_namespace(self.namespace)
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._tokenizer = None
        self._sampling_cls = None
        drop_namespace(self.namespace)
        empty_cuda_cache(collect_ipc=True)
