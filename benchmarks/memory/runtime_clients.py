"""The clients a memory run talks to: the answer backend and the optional judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from benchmarks.common.clients import ChatClient, ChatResult
from benchmarks.common.judge import build_judge_client
from benchmarks.common.vector_types import SearchHit
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, QuestionAnswer


class AnswerClient(Protocol):
    """What the run stage needs from an answer backend."""

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
    """An answer backend that encodes each sample's KV before vLLM starts."""

    def precompute_sample_cache(
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
    answer_client: AnswerClient
    if config.answer_backend == "kv-injection":
        from benchmarks.memory.answer_kv import KVAnswerClient

        answer_client = KVAnswerClient(config)
    else:
        from benchmarks.memory.answer_prompt_injection import PromptInjectionAnswerClient

        answer_client = PromptInjectionAnswerClient(config)
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client_for(config))


def judge_client_for(config: MemoryRunConfig) -> ChatClient | None:
    if config.skip_judge:
        return None
    return build_judge_client(config.judge)
