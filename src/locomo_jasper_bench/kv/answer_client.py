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
from .prompting import build_kv_equivalence_prompt_token_ids
from .strict_gpu_registry import (
    clear_namespace,
    drop_namespace,
    namespace_diagnostics,
    namespace_stats,
    register_user_memory,
    remove_user_memory,
    reset_namespace_diagnostics,
)
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
        self._sample_precompute_metrics: dict[str, Any] = {}

    def precompute_sample_cache(self, sample: ConversationSample) -> None:
        force_vllm_inprocess_mode()
        require_ai_memory_submodule()
        if self._llm is not None:
            raise RuntimeError(
                "GPU-resident KV precompute must run before vLLM is started. "
                "Close vLLM before precomputing the next active sample."
            )
        self.close_sample()
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

        self._composer = composer
        self._active_sample_id = id(sample)
        self._sample_precompute_metrics = composer.cache_stats()
        self._sample_precompute_metrics.update(
            {
                "kv_precompute_time_ms": (time.perf_counter() - started) * 1000,
                "kv_chunk_cache_residency_is_gpu": 1,
            }
        )
        logger.info(
            "Precomputed GPU-resident KV cache sample_id=%s chunks=%s tokens=%s layers=%s gpu_mb=%.1f devices=%s",
            sample.sample_id,
            self._sample_precompute_metrics.get("kv_precomputed_chunks", 0),
            self._sample_precompute_metrics.get("kv_precomputed_tokens", 0),
            self._sample_precompute_metrics.get("kv_precomputed_layers", 0),
            self._sample_precompute_metrics.get("kv_precomputed_gpu_mb", 0.0),
            self._sample_precompute_metrics.get("kv_precomputed_devices", ""),
        )

    def start_llm(self) -> None:
        force_vllm_inprocess_mode()
        if self._llm is not None:
            return
        if self._composer is None:
            raise RuntimeError("GPU-resident KV sample cache must be precomputed before starting vLLM.")

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
            self.close()
            raise

    def prepare_sample(
        self,
        sample: ConversationSample,
        question_hits: list[tuple[QuestionAnswer, list[SearchHit]]] | list[list[SearchHit]],
    ) -> None:
        del question_hits
        if self._composer is None:
            raise RuntimeError(f"GPU-resident KV cache for sample_id={sample.sample_id} was not precomputed.")
        if self._active_sample_id != id(sample):
            raise RuntimeError(f"Strict GPU KV sample_id={sample.sample_id} is not the active precomputed sample.")
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
    ) -> ChatResult:
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
            raise RuntimeError("VLLMChunkedKVAnswerClient.prepare_sample() must be called before answering.")
        composer = self._composer
        if composer is None:
            raise RuntimeError(f"Strict GPU KV cache sample_id={sample.sample_id} was not prepared.")
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
        register_diagnostics = namespace_diagnostics(self.namespace)
        logger.info(
            "Registered strict GPU memory namespace=%s user=%s tokens=%d registry_store_id=%s registry_users=%s",
            self.namespace,
            user_id,
            composed.num_tokens,
            register_diagnostics.get("registry_store_id", 0),
            register_diagnostics.get("registry_user_count", 0),
        )
        try:
            prompt = build_kv_equivalence_prompt_token_ids(
                self._tokenizer,
                list(composed.token_ids),
                sample,
                qa,
            )
            memory_token_ids = list(composed.token_ids)
            local_prefix_match = prompt.prompt_token_ids[: len(memory_token_ids)] == memory_token_ids
            if not local_prefix_match:
                raise RuntimeError(
                    "Strict GPU KV prompt token prefix does not match the registered memory token ids "
                    f"for sample_id={sample.sample_id} question_id={qa.question_id}."
                )
            block_size = _connector_block_size(namespace_diagnostics(self.namespace))
            expected_aligned_tokens = _align_to_block_size(composed.num_tokens, block_size)
            metrics: dict[str, Any] = {
                **self._sample_precompute_metrics,
                "kv_memory_tokens": composed.num_tokens,
                "kv_compose_time_ms": composed.compose_time_ms,
                "kv_context_window": composed.context_window,
                "kv_context_prefix_tokens_total": composed.context_prefix_tokens_total,
                "kv_context_prefix_tokens_max": composed.context_prefix_tokens_max,
                "kv_context_prefix_truncated_tokens": composed.context_prefix_truncated_tokens,
                "kv_prompt_tokens": len(prompt.prompt_token_ids),
                "kv_query_tokens": len(prompt.query_token_ids),
                "kv_query_bos_stripped": int(prompt.stripped_query_bos),
                "kv_local_prefix_match": int(local_prefix_match),
                "kv_cache_block_size": block_size,
                "kv_expected_aligned_tokens": expected_aligned_tokens,
                "kv_uninjected_tail_tokens": composed.num_tokens - expected_aligned_tokens,
            }
            ttft_ms: float | None = None
            ttft_probe_ms = 0.0
            if self.config.measure_ttft:
                reset_namespace_diagnostics(self.namespace)
                _total_ttft_ms, engine_ttft_ms, ttft_probe_ms = self._measure_one_token_ttft(
                    prompt_token_ids=prompt.prompt_token_ids,
                    temperature=temperature,
                    top_p=top_p,
                    request_started=request_started,
                )
                ttft_ms = engine_ttft_ms
                metrics["kv_engine_time_to_first_token_ms"] = engine_ttft_ms
                ttft_diagnostics = namespace_diagnostics(self.namespace)
                metrics.update(_connector_metrics("kv_ttft_connector", ttft_diagnostics))
                _log_connector_diagnostics(
                    "TTFT probe",
                    ttft_diagnostics,
                    expected_aligned_tokens=expected_aligned_tokens,
                )
                _validate_strict_connector_phase(
                    "TTFT probe",
                    ttft_diagnostics,
                    expected_aligned_tokens=expected_aligned_tokens,
                )

            sampling = self._sampling_params(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            reset_namespace_diagnostics(self.namespace)
            generate_started = time.perf_counter()
            outputs = self._llm.generate(
                [{"prompt_token_ids": prompt.prompt_token_ids}],
                sampling,
                use_tqdm=False,
            )
            answer_diagnostics = namespace_diagnostics(self.namespace)
            metrics.update(_connector_metrics("kv_answer_connector", answer_diagnostics))
            _log_connector_diagnostics(
                "answer generation",
                answer_diagnostics,
                expected_aligned_tokens=expected_aligned_tokens,
            )
            _validate_strict_connector_phase(
                "answer generation",
                answer_diagnostics,
                expected_aligned_tokens=expected_aligned_tokens,
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
        if self._composer is not None:
            self._composer.close()
            self._composer = None
        self._active_sample_id = None
        self._sample_precompute_metrics = {}
        clear_namespace(self.namespace)
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def close_llm(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._tokenizer = None
        self._sampling_cls = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    def close(self) -> None:
        self.close_sample()
        self.close_llm()
        drop_namespace(self.namespace)

    def _measure_one_token_ttft(
        self,
        *,
        prompt_token_ids: list[int],
        temperature: float,
        top_p: float,
        request_started: float,
    ) -> tuple[float, float, float]:
        sampling = self._sampling_params(
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

    def _sampling_params(
        self,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        min_tokens: int = 0,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "user": self.active_user_id,
        }
        try:
            return self._sampling_cls(**kwargs)
        except TypeError as exc:
            if "user" not in str(exc):
                raise
            kwargs.pop("user", None)
            return self._sampling_cls(**kwargs)

def _align_to_block_size(num_tokens: int, block_size: int) -> int:
    if block_size <= 0:
        return 0
    return (num_tokens // block_size) * block_size


def _connector_block_size(diagnostics: dict[str, Any]) -> int:
    block_size = int(diagnostics.get("connector_block_size") or 0)
    return block_size if block_size > 0 else 16


def _connector_metrics(prefix: str, diagnostics: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = {
        "connector_init_count": "init_count",
        "connector_match_attempts": "match_attempts",
        "connector_match_hits": "hits",
        "connector_match_misses": "misses",
        "connector_update_state_calls": "update_state_calls",
        "connector_build_meta_calls": "build_meta_calls",
        "connector_metadata_loads": "metadata_loads",
        "connector_start_load_calls": "start_load_calls",
        "connector_injected_tokens": "injected_tokens",
        "connector_missing_memory_loads": "missing_memory_loads",
        "connector_missing_layer_loads": "missing_layer_loads",
        "connector_block_size": "block_size",
        "connector_last_mismatch_index": "last_mismatch_index",
        "connector_last_prompt_tokens": "last_prompt_tokens",
        "connector_last_raw_memory_tokens": "last_raw_memory_tokens",
        "connector_last_aligned_tokens": "last_aligned_tokens",
        "connector_last_new_tokens": "last_new_tokens",
        "connector_last_num_computed_tokens": "last_num_computed_tokens",
        "connector_store_id": "store_id",
        "connector_store_user_count": "store_user_count",
        "registry_store_id": "registry_store_id",
        "registry_user_count": "registry_user_count",
    }
    metrics: dict[str, Any] = {}
    for source_key, metric_key in numeric_keys.items():
        metrics[f"{prefix}_{metric_key}"] = diagnostics.get(source_key, 0)
    metrics[f"{prefix}_last_miss_reason"] = diagnostics.get("connector_last_miss_reason", "")
    metrics[f"{prefix}_last_user_id"] = diagnostics.get("connector_last_user_id", "")
    metrics[f"{prefix}_last_request_id"] = diagnostics.get("connector_last_request_id", "")
    metrics[f"{prefix}_last_role"] = diagnostics.get("connector_last_role", "")
    return metrics


def _log_connector_diagnostics(
    phase: str,
    diagnostics: dict[str, Any],
    *,
    expected_aligned_tokens: int,
) -> None:
    logger.info(
        "Strict GPU KV %s diagnostics: attempts=%s hits=%s misses=%s injected_tokens=%s "
        "expected_aligned_tokens=%s miss_reason=%s mismatch_index=%s prompt_tokens=%s memory_tokens=%s "
        "user=%s connector_store_id=%s connector_users=%s registry_store_id=%s registry_users=%s",
        phase,
        diagnostics.get("connector_match_attempts", 0),
        diagnostics.get("connector_match_hits", 0),
        diagnostics.get("connector_match_misses", 0),
        diagnostics.get("connector_injected_tokens", 0),
        expected_aligned_tokens,
        diagnostics.get("connector_last_miss_reason", ""),
        diagnostics.get("connector_last_mismatch_index", -1),
        diagnostics.get("connector_last_prompt_tokens", 0),
        diagnostics.get("connector_last_raw_memory_tokens", 0),
        diagnostics.get("connector_last_user_id", ""),
        diagnostics.get("connector_store_id", 0),
        diagnostics.get("connector_store_user_count", 0),
        diagnostics.get("registry_store_id", 0),
        diagnostics.get("registry_user_count", 0),
    )


def _validate_strict_connector_phase(
    phase: str,
    diagnostics: dict[str, Any],
    *,
    expected_aligned_tokens: int,
) -> None:
    if expected_aligned_tokens <= 0:
        raise RuntimeError(
            f"Strict GPU KV {phase} cannot inject memory because the block-aligned memory token count is "
            f"{expected_aligned_tokens}. Increase retrieved memory tokens or check the vLLM block size."
        )

    attempts = int(diagnostics.get("connector_match_attempts") or 0)
    hits = int(diagnostics.get("connector_match_hits") or 0)
    injected_tokens = int(diagnostics.get("connector_injected_tokens") or 0)
    if attempts <= 0:
        raise RuntimeError(
            f"Strict GPU KV connector was not consulted during {phase}. "
            "vLLM is prefilling the full prompt instead of using injected KV."
        )
    if hits <= 0:
        reason = diagnostics.get("connector_last_miss_reason") or "unknown"
        user_id = diagnostics.get("connector_last_user_id") or ""
        mismatch_index = diagnostics.get("connector_last_mismatch_index", -1)
        prompt_tokens = diagnostics.get("connector_last_prompt_tokens", 0)
        raw_tokens = diagnostics.get("connector_last_raw_memory_tokens", 0)
        connector_store_id = diagnostics.get("connector_store_id", 0)
        registry_store_id = diagnostics.get("registry_store_id", 0)
        connector_users = diagnostics.get("connector_store_user_count", 0)
        registry_users = diagnostics.get("registry_user_count", 0)
        raise RuntimeError(
            f"Strict GPU KV connector did not match during {phase}: reason={reason} "
            f"user={user_id} mismatch_index={mismatch_index} prompt_tokens={prompt_tokens} "
            f"memory_tokens={raw_tokens} connector_store_id={connector_store_id} "
            f"connector_users={connector_users} registry_store_id={registry_store_id} "
            f"registry_users={registry_users}."
        )
    if injected_tokens < expected_aligned_tokens:
        raise RuntimeError(
            f"Strict GPU KV connector matched during {phase} but injected only {injected_tokens} tokens; "
            f"expected at least {expected_aligned_tokens}."
        )


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
