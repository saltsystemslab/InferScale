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
from .strict_gpu_registry import clear_namespace, register_user_memory, remove_user_memory
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
        self._composers: dict[int, ChunkedRopeSampleComposer] = {}

    def prepare_sample(self, sample: ConversationSample, hits_by_question: list[list[SearchHit]]) -> None:
        self.prepare_samples([(sample, hits_by_question)])

    def prepare_samples(
        self,
        samples: list[tuple[ConversationSample, list[list[SearchHit]]]],
    ) -> None:
        if not samples:
            raise RuntimeError("Strict GPU KV mode cannot prepare an empty sample window.")
        if self.config.kv_sample_window < 1:
            raise RuntimeError("--kv-sample-window must be >= 1.")
        if len(samples) > self.config.kv_sample_window:
            raise RuntimeError(
                f"Strict GPU KV sample window got {len(samples)} samples, "
                f"exceeding --kv-sample-window {self.config.kv_sample_window}."
            )

        force_vllm_inprocess_mode()
        self.close_sample()
        require_ai_memory_submodule()

        try:
            for sample, hits_by_question in samples:
                needed_turn_ids: set[str] = set()
                for hits in hits_by_question:
                    needed_turn_ids.update(selected_turn_ids(hits))
                if not needed_turn_ids:
                    raise RuntimeError(
                        f"No retrieved turn ids for sample_id={sample.sample_id}; cannot prepare KV memory."
                    )

                logger.info(
                    "Preparing strict GPU KV sample_id=%s retrieved_turns=%d window=%d/%d",
                    sample.sample_id,
                    len(needed_turn_ids),
                    len(self._composers) + 1,
                    len(samples),
                )
                composer = ChunkedRopeSampleComposer(
                    model=self.config.model,
                    dtype=self.config.kv_dtype,
                    device=self.config.kv_device,
                    max_position=self.config.kv_max_position,
                )
                try:
                    composer.encode_sample(sample, turn_ids=needed_turn_ids)
                    self._free_composer_encoder(composer)
                except Exception:
                    composer.close()
                    raise
                self._composers[id(sample)] = composer
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
    ) -> ChatResult:
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
            raise RuntimeError("VLLMChunkedKVAnswerClient.prepare_sample() must be called before answering.")
        composer = self._composers.get(id(sample))
        if composer is None:
            raise RuntimeError(
                f"Strict GPU KV sample_id={sample.sample_id} was not prepared in the active sample window."
            )

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
            text = outputs[0].outputs[0].text.strip()
            return ChatResult(
                content=text,
                ttft_ms=ttft_ms,
            )
        finally:
            remove_user_memory(self.namespace, user_id)

    def close_sample(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        for composer in self._composers.values():
            composer.close()
        self._composers.clear()
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
