from __future__ import annotations

import gc
import logging
import os
import time
import uuid
from typing import Any

from ..clients import ChatResult
from ..config import BenchmarkConfig
from ..data import ConversationSample, QuestionAnswer
from ..vector_types import SearchHit
from .chunked_rope import ChunkedRopeSampleComposer, selected_turn_ids
from .strict_gpu_registry import clear_namespace, namespace_stats, register_user_memory, remove_user_memory
from .submodule import require_ai_memory_submodule

logger = logging.getLogger(__name__)


class VLLMChunkedKVAnswerClient:
    """In-process vLLM answer client using strict GPU chunked-RoPE KV injection."""

    def __init__(self, config: BenchmarkConfig) -> None:
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

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
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
                    composition_mode=self.config.kv_composition_mode,
                )
                try:
                    composer.encode_sample(sample, turn_ids=needed_turn_ids)
                    composer.precompose_contiguous(hits_by_question)
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
        ttft_started_at: float | None = None,
    ) -> ChatResult:
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
            raise RuntimeError("VLLMChunkedKVAnswerClient.prepare_sample() must be called before answering.")
        composer = self._composers.get(id(sample))
        if composer is None:
            raise RuntimeError(
                f"Strict GPU KV sample_id={sample.sample_id} was not prepared in the active sample window."
            )

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
            query_tokens = self._query_token_ids(sample, qa)
            prompt_token_ids = list(composed.token_ids) + query_tokens
            prompt_bos_count = count_bos_tokens(self._tokenizer, prompt_token_ids)
            sampling = self._sampling_cls(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            generate_started = time.perf_counter()
            outputs = self._llm.generate(
                [{"prompt_token_ids": prompt_token_ids}],
                sampling,
                use_tqdm=False,
            )
            finished = time.perf_counter()
            generate_ms = (finished - generate_started) * 1000
            total_ms = (finished - request_started) * 1000
            text = outputs[0].outputs[0].text.strip()
            stats = namespace_stats(self.namespace)
            return ChatResult(
                content=text,
                metrics={
                    "kv_memory_tokens": composed.num_tokens,
                    "kv_compose_time_ms": composed.compose_time_ms,
                    "answer_generate_time_ms": generate_ms,
                    "answer_total_time_ms": total_ms,
                    "kv_query_tokens": len(query_tokens),
                    "kv_prompt_bos_count": prompt_bos_count,
                    "kv_injected_prefix_tokens": composed.num_tokens,
                    "kv_composition_mode": self.config.kv_composition_mode,
                    "kv_store_gpu_mb": stats.get("total_gpu_mb", 0.0),
                    "kv_selected_turn_ids": composed.selected_turn_ids,
                },
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

    def _query_token_ids(self, sample: ConversationSample, qa: QuestionAnswer) -> list[int]:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Conversation id: {sample.sample_id}\n\n"
                    f"Question: {qa.question}\n\n"
                    "Answer:"
                ),
            },
        ]
        return tokenize_messages(self._tokenizer, messages, strip_leading_bos=True)


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


def tokenize_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    strip_leading_bos: bool = False,
) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        token_ids = list(
            apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        return strip_leading_bos_token(tokenizer, token_ids) if strip_leading_bos else token_ids

    text = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Tokenizer has neither apply_chat_template nor encode.")
    token_ids = list(encode(text))
    return strip_leading_bos_token(tokenizer, token_ids) if strip_leading_bos else token_ids


def strip_leading_bos_token(tokenizer: Any, token_ids: list[int]) -> list[int]:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None and token_ids and token_ids[0] == bos_token_id:
        return token_ids[1:]
    return token_ids


def count_bos_tokens(tokenizer: Any, token_ids: list[int]) -> int:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        return 0
    return sum(1 for token_id in token_ids if token_id == bos_token_id)


def _memory_user_id(sample_id: str, question_id: str) -> str:
    safe_sample = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in sample_id)
    safe_question = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in question_id)
    return f"{safe_sample}__{safe_question}"
