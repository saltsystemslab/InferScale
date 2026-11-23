from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from benchmarks.common.clients import ChatClient, ChatResult
from benchmarks.common.judge import build_judge_client
from benchmarks.common.vector_types import SearchHit
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, QuestionAnswer
from benchmarks.memory.protocol import LMCACHE_BACKEND


class AnswerClient(Protocol):
    def start_llm(self) -> None: ...

    def prepare_sample(self, sample: ConversationSample) -> None: ...

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
    ) -> ChatResult: ...

    def close_sample(self) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class KVPrecomputeClient(Protocol):
    def precompute_sample_cache(
        self, sample: ConversationSample, facts: list[SearchHit]
    ) -> dict[str, Any]: ...


@runtime_checkable
class MemoryWarmupClient(Protocol):
    def warm_sample_memory(
        self, sample: ConversationSample, facts: list[SearchHit]
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class RuntimeClients:
    answer_client: AnswerClient
    judge_client: ChatClient | None


def build_clients(config: MemoryRunConfig) -> RuntimeClients:
    logger.info(
        "Configuring clients answer_backend={} judge_provider={} judge_endpoint={}",
        config.answer_backend,
        config.judge_provider,
        config.judge_base_url,
    )
    answer_client = _build_answer_client(config)
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client_for(config))


def _build_answer_client(config: MemoryRunConfig) -> AnswerClient:
    if config.answer_backend == "kv-injection":
        from benchmarks.memory.answer_kv import KVAnswerClient

        return KVAnswerClient(config)
    if config.answer_backend == "prompt-injection":
        from benchmarks.memory.answer_prompt_injection import PromptInjectionAnswerClient

        return PromptInjectionAnswerClient(config)
    if config.answer_backend == LMCACHE_BACKEND:
        from benchmarks.memory.answer_lmcache import LMCacheBlendAnswerClient

        return LMCacheBlendAnswerClient(config)
    raise ValueError(f"Unsupported answer backend: {config.answer_backend!r}.")


def judge_client_for(config: MemoryRunConfig) -> ChatClient | None:
    if config.skip_judge:
        return None
    return build_judge_client(config.judge)
