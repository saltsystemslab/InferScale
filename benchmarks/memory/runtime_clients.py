from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from benchmarks.common.clients import ChatClient
from benchmarks.common.judge import build_judge_client
from benchmarks.memory.config import MemoryRunConfig


@dataclass(slots=True)
class RuntimeClients:
    answer_client: Any
    judge_client: ChatClient | None


def build_clients(config: MemoryRunConfig) -> RuntimeClients:
    logger.info(
        "Configuring clients answer_backend={} judge_provider={} judge_endpoint={}",
        config.answer_backend,
        config.judge_provider,
        config.judge_base_url,
    )
    if config.answer_backend == "kv-injection":
        from benchmarks.memory.answer_kv import KVAnswerClient

        answer_client = KVAnswerClient(config)
    else:
        from benchmarks.memory.answer_prefix import PrefixAnswerClient

        answer_client = PrefixAnswerClient(config)
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client_for(config))


def judge_client_for(config: MemoryRunConfig) -> ChatClient | None:
    if config.skip_judge:
        return None
    return build_judge_client(
        provider=config.judge_provider,
        model=config.judge_model,
        base_url=config.judge_base_url,
        api_key=config.judge_api_key,
    )
