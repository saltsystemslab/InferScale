from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from .clients import ChatClient, OpenAICompatibleChatClient, OpenAIResponsesJudgeClient
from .config import BenchmarkConfig


@dataclass(slots=True)
class RuntimeClients:
    answer_client: Any
    judge_client: ChatClient | None


def build_clients(config: BenchmarkConfig) -> RuntimeClients:
    logger.info(
        "Configuring clients answer_backend={} judge_provider={} judge_endpoint={}",
        config.answer_backend,
        config.judge_provider,
        config.judge_base_url,
    )
    if config.answer_backend == "vllm-kv":
        from .kv.answer_client import VLLMChunkedKVAnswerClient

        answer_client = VLLMChunkedKVAnswerClient(config)
    else:
        from .kv.prefix_answer_client import VLLMPrefixPromptAnswerClient

        answer_client = VLLMPrefixPromptAnswerClient(config)
    judge_client = build_judge_client(config)
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client)


def build_judge_client(config: BenchmarkConfig) -> ChatClient | None:
    if config.skip_judge or config.judge_provider == "none":
        return None
    if config.judge_provider == "openai":
        if not config.judge_api_key:
            raise RuntimeError("OPENAI_API_KEY is required to use --judge openai.")
        return OpenAIResponsesJudgeClient(
            api_key=config.judge_api_key,
            base_url=config.judge_base_url,
            model=config.judge_model,
        )
    if config.judge_provider == "vllm":
        if not config.judge_base_url:
            raise RuntimeError("JUDGE_BASE_URL is required to use --judge vllm.")
        if not config.judge_api_key:
            raise RuntimeError("JUDGE_API_KEY is required to use --judge vllm.")
        return OpenAICompatibleChatClient(
            base_url=config.judge_base_url,
            api_key=config.judge_api_key,
            model=config.judge_model,
        )
    raise RuntimeError(f"Unsupported judge provider: {config.judge_provider}")
