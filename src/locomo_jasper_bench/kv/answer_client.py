from __future__ import annotations

import gc
import logging
import os
import sys
import time
import uuid
from typing import Any

from ..clients import ChatResult
from ..config import BenchmarkConfig
from ..data import ConversationSample, QuestionAnswer
from ..vector_types import SearchHit
from .chunked_rope import ChunkedRopeSampleComposer
from .prompting import build_kv_equivalence_prompt_token_ids, selected_turn_ids
from .strict_gpu_registry import clear_namespace, namespace_stats, register_user_memory, remove_user_memory
from .submodule import require_ai_memory_submodule

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
        self._composer: ChunkedRopeSampleComposer | None = None
        self._active_sample_id: int | None = None

    def prepare_sample(self, sample: ConversationSample, hits_by_question: list[list[SearchHit]]) -> None:
        force_vllm_inprocess_mode()
        self.close_sample()
        require_ai_memory_submodule()

        try:
            needed_turn_ids: set[str] = set()
            for hits in hits_by_question:
                needed_turn_ids.update(selected_turn_ids(hits))
            if not needed_turn_ids:
                raise RuntimeError(f"No retrieved turn ids for sample_id={sample.sample_id}; cannot prepare KV memory.")

            logger.info(
                "Preparing strict GPU KV sample_id=%s retrieved_turns=%d context_window=%d",
                sample.sample_id,
                len(needed_turn_ids),
                self.config.context_window,
            )
            composer = ChunkedRopeSampleComposer(
                model=self.config.model,
                dtype=self.config.kv_dtype,
                device=self.config.kv_device,
                max_position=self.config.kv_max_position,
                context_window=self.config.context_window,
            )
            try:
                composer.encode_sample(sample, turn_ids=needed_turn_ids)
                self._free_composer_encoder(composer)
            except Exception:
                composer.close()
                raise
            self._composer = composer
            self._active_sample_id = id(sample)
        except Exception:
            self.close_sample()
            raise

        from vllm import LLM, SamplingParams

        try:
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
                kv_transfer_config=build_strict_gpu_kv_transfer_config(
                    connector_module=self.config.kv_connector_module,
                    namespace=self.namespace,
                    default_user_id=self.active_user_id,
                ),
            )
            self._tokenizer = self._llm.get_tokenizer()
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
        ttft_started_at: float | None = None,
    ) -> ChatResult:
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
            raise RuntimeError("VLLMChunkedKVAnswerClient.prepare_sample() must be called before answering.")
        composer = self._composer
        if composer is None:
            raise RuntimeError(f"Strict GPU KV sample_id={sample.sample_id} was not prepared.")
        if self._active_sample_id != id(sample):
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
                "kv_memory_tokens": composed.num_tokens,
                "kv_compose_time_ms": composed.compose_time_ms,
                "kv_context_window": composed.context_window,
                "kv_context_prefix_tokens_total": composed.context_prefix_tokens_total,
                "kv_context_prefix_tokens_max": composed.context_prefix_tokens_max,
                "kv_context_prefix_truncated_tokens": composed.context_prefix_truncated_tokens,
                "kv_query_tokens": len(prompt.query_token_ids),
                "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            }
            ttft_ms: float | None = None
            ttft_probe_ms = 0.0
            if self.config.measure_ttft:
                _total_ttft_ms, engine_ttft_ms, ttft_probe_ms = self._measure_one_token_ttft(
                    prompt_token_ids=prompt.prompt_token_ids,
                    temperature=temperature,
                    top_p=top_p,
                    request_started=request_started,
                )
                ttft_ms = engine_ttft_ms
                metrics["kv_engine_time_to_first_token_ms"] = engine_ttft_ms

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
            total_ms = max(0.0, (finished - request_started) * 1000 - ttft_probe_ms)
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
        if self._llm is not None:
            del self._llm
            self._llm = None
        if self._composer is not None:
            self._composer.close()
            self._composer = None
        self._active_sample_id = None
        self._tokenizer = None
        clear_namespace(self.namespace)
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def close(self) -> None:
        self.close_sample()

    def _measure_one_token_ttft(
        self,
        *,
        prompt_token_ids: list[int],
        temperature: float,
        top_p: float,
        request_started: float,
    ) -> tuple[float, float, float]:
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
        return (
            (finished - request_started) * 1000,
            (finished - engine_started) * 1000,
            (finished - engine_started) * 1000,
        )

    @staticmethod
    def _free_composer_encoder(composer: ChunkedRopeSampleComposer) -> None:
        # The encoded chunks stay GPU-resident; the HF model is released before vLLM loads.
        composer.encoder._model = None
        composer.hf_model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

def build_strict_gpu_kv_transfer_config(
    *,
    connector_module: str,
    namespace: str,
    default_user_id: str = "default",
) -> dict[str, Any]:
    return {
        "kv_connector": "MemoryKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": connector_module,
        "kv_connector_extra_config": {
            "memory_namespace": namespace,
            "default_user_id": default_user_id,
        },
    }


def force_vllm_inprocess_mode() -> None:
    """Force vLLM V1 offline LLM execution into this process.

    The strict GPU connector reads from a process-local registry populated by
    the benchmark process. If vLLM starts an EngineCore subprocess, that
    registry is not shared with the connector.
    """
    current = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if _env_truthy(current):
        logger.warning(
            "Overriding VLLM_ENABLE_V1_MULTIPROCESSING=%s because strict GPU KV mode requires "
            "vLLM's offline engine to share this process.",
            current,
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    loaded_envs = sys.modules.get("vllm.envs")
    if loaded_envs is not None and hasattr(loaded_envs, "VLLM_ENABLE_V1_MULTIPROCESSING"):
        setattr(loaded_envs, "VLLM_ENABLE_V1_MULTIPROCESSING", False)


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _memory_user_id(sample_id: str, question_id: str) -> str:
    safe_sample = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sample_id)
    safe_question = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in question_id)
    return f"{safe_sample}__{safe_question}"
