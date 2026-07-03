from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from .clients import ChatClient, OpenAICompatibleChatClient
from .config import BenchmarkConfig


@dataclass(slots=True)
class RuntimeClients:
    answer_client: Any
    judge_client: ChatClient | None


def build_clients(config: BenchmarkConfig) -> RuntimeClients:
    logger.info(
        "Configuring clients answer_backend={} judge={}",
        config.answer_backend,
        config.judge_base_url,
    )
    if config.answer_backend == "vllm-kv":
        from .kv.answer_client import VLLMChunkedKVAnswerClient

        answer_client = VLLMChunkedKVAnswerClient(config)
    else:
        from .kv.prefix_answer_client import VLLMPrefixPromptAnswerClient

        answer_client = VLLMPrefixPromptAnswerClient(config)
    if config.skip_judge:
        judge_client = None
    else:
        judge_client = OpenAICompatibleChatClient(
            base_url=config.judge_base_url,
            api_key=config.judge_api_key,
            model=config.judge_model,
        )
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client)
