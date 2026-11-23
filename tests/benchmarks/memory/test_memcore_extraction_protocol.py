from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from memory_config import make_memory_config

from benchmarks.memory.data import ConversationSample, QuestionAnswer, Turn
from benchmarks.memory.judging import build_judge_messages
from benchmarks.memory.mem0.fact_catalog import FactCatalogStore, make_memory_fact
from benchmarks.memory.mem0.llm_cache import CachedMemoryLLM
from benchmarks.memory.protocol import (
    JUDGE_PROMPT_PROTOCOL,
    MEMORY_EXTRACTION_RETRY_TEMPERATURES,
    MEMORY_EXTRACTION_RETRY_TOP_P,
    MEMORY_LLM_TEMPERATURE,
)

MESSAGES = [{"role": "user", "content": "Remember blue."}]
AS_EXTRACTION = {"response_format": {"type": "json_object"}}
INVALID_EXTRACTION = '{"memory":[{"id":"0"}]'
VALID_EXTRACTION = json.dumps(
    {
        "memory": [
            {
                "id": "0",
                "text": "User likes blue.",
                "attributed_to": "user",
                "linked_memory_ids": [],
            }
        ]
    },
    separators=(",", ":"),
)
WARMER_TEMPERATURE = 0.2


class MemoryLLM:
    provider_name = "recording"

    def __init__(self, responses: Iterable[str] = ()) -> None:
        self.responses = iter(responses)
        self.sampling: list[dict[str, float]] = []

    def generate_response(self, messages: Any, *args: Any, **options: Any) -> str:
        self.sampling.append(
            {name: options[name] for name in ("temperature", "top_p") if name in options}
        )
        return next(self.responses)


def cached(llm: MemoryLLM, cache_dir: Path, mode: str, **identity: Any) -> CachedMemoryLLM:
    return CachedMemoryLLM(llm, cache_dir, "vllm", "model", mode, **identity)


def conversation() -> ConversationSample:
    turn = Turn("conv-1", "session_1", 1, 0, "Alice", "I like tea.", "2026-01-02")
    return ConversationSample("conv-1", [turn], [], {})


def catalog_store(root: Path, **identity: Any) -> FactCatalogStore:
    return FactCatalogStore(
        root,
        provider="vllm",
        model="model",
        endpoint="http://localhost:8000/v1",
        embedding_model="embedding-model",
        embedding_endpoint=None,
        mem0_version="2.0.11",
        **identity,
    )


def test_fact_catalog_extracted_at_another_temperature_is_never_replayed(tmp_path):
    sample = conversation()
    fact = make_memory_fact("Alice likes tea.", sample, sample.turns[0])
    catalog_store(tmp_path, temperature=MEMORY_LLM_TEMPERATURE).write(sample, [fact])

    assert catalog_store(tmp_path, temperature=MEMORY_LLM_TEMPERATURE).load(sample) == (fact,)
    with pytest.raises(RuntimeError, match="identity mismatch for memory_llm_temperature"):
        catalog_store(tmp_path, temperature=WARMER_TEMPERATURE).load(sample)


def test_first_extraction_attempt_keeps_the_baseline_sampling_and_each_retry_is_warmer(tmp_path):
    llm = MemoryLLM([INVALID_EXTRACTION, INVALID_EXTRACTION, VALID_EXTRACTION])

    response = cached(llm, tmp_path, "write").generate_response(MESSAGES, **AS_EXTRACTION)

    assert response == VALID_EXTRACTION
    assert llm.sampling == [
        {},
        {"temperature": 0.2, "top_p": MEMORY_EXTRACTION_RETRY_TOP_P},
        {"temperature": 0.5, "top_p": MEMORY_EXTRACTION_RETRY_TOP_P},
    ]
    assert MEMORY_EXTRACTION_RETRY_TEMPERATURES == (0.2, 0.5)


def test_extraction_retries_widen_the_narrow_top_p_that_mem0_sends_by_default():
    from mem0.configs.llms.vllm import VllmConfig

    assert VllmConfig().top_p == 0.1
    assert MEMORY_EXTRACTION_RETRY_TOP_P == 1.0


def test_extraction_retries_do_not_change_the_cache_identity(tmp_path):
    writer = cached(MemoryLLM([INVALID_EXTRACTION, VALID_EXTRACTION]), tmp_path, "write")
    writer.generate_response(MESSAGES, **AS_EXTRACTION)
    reader_llm = MemoryLLM()
    reader = cached(reader_llm, tmp_path, "read")

    assert reader.generate_response(MESSAGES, **AS_EXTRACTION) == VALID_EXTRACTION
    assert reader_llm.sampling == []


def test_judge_answers_one_boolean_word_under_the_upstream_json_scoring_protocol_name():
    question = QuestionAnswer("conv-1", "q0", "Where did Alice go?", "Lyon", "1")
    config = make_memory_config()

    (message,) = build_judge_messages(question, "Lyon")

    assert JUDGE_PROMPT_PROTOCOL == "memory-benchmarks-json-jscore-v1"
    assert "Answer with exactly one lowercase word: true or false." in message["content"]
    assert "Do not return JSON" in message["content"]
    assert config.max_judge_tokens == 4
    assert config.to_jsonable()["judge_prompt_protocol"] == JUDGE_PROMPT_PROTOCOL
