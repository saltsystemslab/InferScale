from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from ..clients import ChatResult
from ..config import BenchmarkConfig
from ..data import ConversationSample, QuestionAnswer
from ..vector_types import SearchHit
from .chunked_rope import (
    CachedChunkedRopeSampleComposer,
    ChunkedRopeSampleComposer,
    load_sample_kv_cache,
    save_sample_kv_cache,
)
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
        self._composer: CachedChunkedRopeSampleComposer | None = None
        self._active_sample_id: int | None = None
        self._sample_cache_dir: Path | None = None

    def prepare_sample(
        self,
        sample: ConversationSample,
        question_hits: list[tuple[QuestionAnswer, list[SearchHit]]] | list[list[SearchHit]],
    ) -> None:
        force_vllm_inprocess_mode()
        self.close_sample()
        require_ai_memory_submodule()

        try:
            hit_lists = _prepared_hit_lists(question_hits)
            needed_turn_ids: set[str] = set()
            for hits in hit_lists:
                needed_turn_ids.update(selected_turn_ids(hits))
            if not needed_turn_ids:
                raise RuntimeError(f"No retrieved turn ids for sample_id={sample.sample_id}; cannot prepare KV memory.")

            cache_dir = self._sample_cache_dir_for(sample)
            cache_path = cache_dir / "sample_kv.pt"
            logger.info(
                "Precomputing CPU KV cache sample_id=%s turns=%d retrieved_turns=%d context_window=%d",
                sample.sample_id,
                len(sample.turns),
                len(needed_turn_ids),
                self.config.context_window,
            )
            _build_sample_kv_cache_in_subprocess(
                self.config,
                sample,
                cache_path,
            )
            self._composer = load_sample_kv_cache(cache_path, device=self.config.kv_device)
            self._sample_cache_dir = cache_dir
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
            raise RuntimeError(f"Strict GPU KV CPU cache sample_id={sample.sample_id} was not prepared.")
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
        if self._sample_cache_dir is not None:
            shutil.rmtree(self._sample_cache_dir, ignore_errors=True)
            self._sample_cache_dir = None
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

    def _sample_cache_dir_for(self, sample: ConversationSample) -> Path:
        return self.config.run_dir / "kv_chunk_cache" / _safe_path_component(sample.sample_id)


def _build_sample_kv_cache_in_subprocess(
    config: BenchmarkConfig,
    sample: ConversationSample,
    cache_path: Path,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = cache_path.with_suffix(".error.txt")
    for path in (cache_path, error_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    ctx = mp.get_context("spawn")
    process = ctx.Process(
        target=_build_sample_kv_cache_worker,
        args=(config, sample, str(cache_path), str(error_path)),
        name=f"kv-cache-{_safe_path_component(sample.sample_id)}",
    )
    process.start()
    process.join()

    if process.exitcode != 0:
        details = ""
        if error_path.exists():
            details = error_path.read_text()
        raise RuntimeError(
            f"KV cache encoder process failed for sample_id={sample.sample_id} "
            f"with exit code {process.exitcode}.\n{details}"
        )
    if not cache_path.exists():
        raise RuntimeError(f"KV cache encoder process did not write {cache_path}.")


def _build_sample_kv_cache_worker(
    config: BenchmarkConfig,
    sample: ConversationSample,
    cache_path: str,
    error_path: str,
) -> None:
    composer: ChunkedRopeSampleComposer | None = None
    try:
        force_vllm_inprocess_mode()
        require_ai_memory_submodule()
        composer = ChunkedRopeSampleComposer(
            model=config.model,
            dtype=config.kv_dtype,
            device=config.kv_device,
            max_position=config.kv_max_position,
            context_window=config.context_window,
        )
        composer.encode_sample(sample)
        save_sample_kv_cache(Path(cache_path), composer)
    except BaseException:
        Path(error_path).write_text(traceback.format_exc())
        raise
    finally:
        if composer is not None:
            composer.close()


def _prepared_hit_lists(
    question_hits: list[tuple[QuestionAnswer, list[SearchHit]]] | list[list[SearchHit]],
) -> list[list[SearchHit]]:
    hit_lists: list[list[SearchHit]] = []
    for item in question_hits:
        if isinstance(item, tuple) and len(item) == 2:
            hit_lists.append(item[1])
        else:
            hit_lists.append(item)
    return hit_lists


def _safe_path_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    return safe or "sample"


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
