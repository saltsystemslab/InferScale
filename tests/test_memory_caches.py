from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from typing import Any

import numpy as np
import pytest

from locomo_jasper_bench.embedding.cache import CachedEmbedder, CachedEmbeddingMissingError
from locomo_jasper_bench.protocol import (
    MEMORY_EXTRACTION_MAX_FACTS,
    MEMORY_EXTRACTION_MAX_MODEL_LEN,
    MEMORY_EXTRACTION_MAX_TEXT_CHARS,
    MEMORY_EXTRACTION_MAX_TOKENS,
    MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
    MEMORY_EXTRACTION_RETRY_TEMPERATURES,
)
from locomo_jasper_bench.retrieval.memory_llm_cache import CachedMemoryLLM, CachedMemoryLLMMissingError
from locomo_jasper_bench.retrieval.memory_llm_protocol import (
    InvalidMemoryExtractionResponseError,
    validate_memory_extraction_response,
)


class RecordingEmbedder:
    def __init__(self) -> None:
        self.batch_calls: list[tuple[list[Any], tuple[Any, ...], dict[str, Any]]] = []
        self.scalar_calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self.provider_name = "recording"

    def embed(self, text: Any, *args: Any, **kwargs: Any) -> list[float]:
        self.scalar_calls.append((text, args, kwargs))
        return _vector_for(text)

    def embed_batch(self, texts: list[Any], *args: Any, **kwargs: Any) -> list[list[float]]:
        self.batch_calls.append((list(texts), args, kwargs))
        return [_vector_for(text) for text in texts]


class RecordingMemoryLLM:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self.provider_name = "recording"

    def generate_response(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, args, kwargs))
        return self.response


class SequenceMemoryLLM:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self.provider_name = "sequence"

    def generate_response(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((messages, args, kwargs))
        return self.responses[len(self.calls) - 1]


def _vector_for(value: Any) -> list[float]:
    text = str(value)
    return [float(len(text)), float(sum(text.encode("utf-8")))]


def _valid_extraction_response(text: str = "User likes blue.") -> str:
    return json.dumps(
        {
            "memory": [
                {
                    "id": "0",
                    "text": text,
                    "attributed_to": "user",
                    "linked_memory_ids": [],
                }
            ]
        },
        separators=(",", ":"),
    )


def test_cached_embedder_batch_uses_partial_hits_and_preserves_order(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    seed_wrapped = RecordingEmbedder()
    seed = CachedEmbedder(seed_wrapped, cache_dir=cache_dir, model="test/model", mode="write")
    assert seed.embed_batch(["cached"], "add") == [_vector_for("cached")]

    wrapped = RecordingEmbedder()
    cached = CachedEmbedder(wrapped, cache_dir=cache_dir, model="test/model", mode="write")
    result = cached.embed_batch(["cached", "new", "cached", "other", "new"], "add")

    assert result == [
        _vector_for("cached"),
        _vector_for("new"),
        _vector_for("cached"),
        _vector_for("other"),
        _vector_for("new"),
    ]
    assert wrapped.batch_calls == [(["new", "other"], ("add",), {})]
    assert wrapped.scalar_calls == []
    assert cached.stats()["hits"] == 2
    assert cached.stats()["misses"] == 3


def test_cached_embedder_batch_read_mode_never_calls_wrapped_embedder(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    writer = CachedEmbedder(RecordingEmbedder(), cache_dir=cache_dir, model="model", mode="write")
    writer.embed_batch(["first", "second"], memory_action="add")

    wrapped = RecordingEmbedder()
    reader = CachedEmbedder(wrapped, cache_dir=cache_dir, model="model", mode="read")
    assert reader.embed_batch(["second", "first", "second"], memory_action="add") == [
        _vector_for("second"),
        _vector_for("first"),
        _vector_for("second"),
    ]
    with pytest.raises(CachedEmbeddingMissingError, match="--preembed-only"):
        reader.embed_batch(["first", "missing"], memory_action="add")

    assert wrapped.batch_calls == []
    assert wrapped.scalar_calls == []


def test_cached_embedder_batch_read_mode_rejects_corrupt_entries_without_fallback(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    writer = CachedEmbedder(RecordingEmbedder(), cache_dir=cache_dir, model="model", mode="write")
    writer.embed_batch(["value"], "add")
    cache_file = next(writer.cache_dir.glob("*.npy"))
    cache_file.write_text("not a numpy file", encoding="utf-8")

    wrapped = RecordingEmbedder()
    reader = CachedEmbedder(wrapped, cache_dir=cache_dir, model="model", mode="read")
    with pytest.raises(CachedEmbeddingMissingError, match="--preembed-only"):
        reader.embed_batch(["value"], "add")

    assert wrapped.batch_calls == []
    assert wrapped.scalar_calls == []


def test_cached_embedder_concurrent_same_key_writes_are_atomic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)
    original_save = np.save

    def synchronized_save(*args: Any, **kwargs: Any) -> None:
        barrier.wait(timeout=5)
        original_save(*args, **kwargs)

    monkeypatch.setattr("locomo_jasper_bench.embedding.cache.np.save", synchronized_save)
    first = CachedEmbedder(RecordingEmbedder(), cache_dir=tmp_path, model="model", mode="write")
    second = CachedEmbedder(RecordingEmbedder(), cache_dir=tmp_path, model="model", mode="write")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda cache: cache.embed("same", "add"), (first, second)))

    assert results == [_vector_for("same"), _vector_for("same")]
    cache_files = list(first.cache_dir.glob("*.npy"))
    assert len(cache_files) == 1
    assert np.load(cache_files[0]).tolist() == _vector_for("same")
    assert list(first.cache_dir.glob("*.tmp")) == []


def test_cached_memory_llm_replays_canonical_requests_and_delegates_attributes(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    response = _valid_extraction_response()
    wrapped = RecordingMemoryLLM(response)
    writer = CachedMemoryLLM(wrapped, cache_dir, "openai", "gpt-5-mini", "write")

    assert writer.generate_response(messages, response_format={"type": "json_object"}, temperature=0) == response
    assert len(wrapped.calls) == 1
    effective_kwargs = wrapped.calls[0][2]
    assert effective_kwargs["max_tokens"] == MEMORY_EXTRACTION_MAX_TOKENS
    schema = effective_kwargs["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["memory"]["maxItems"] == MEMORY_EXTRACTION_MAX_FACTS
    assert (
        schema["properties"]["memory"]["items"]["properties"]["text"]["maxLength"]
        == MEMORY_EXTRACTION_MAX_TEXT_CHARS
    )
    assert writer.provider_name == "recording"
    assert writer.stats()["misses"] == 1

    replay_wrapped = RecordingMemoryLLM("must not be returned")
    reader = CachedMemoryLLM(replay_wrapped, cache_dir, "openai", "gpt-5-mini", "read")
    replayed = reader.generate_response(messages, temperature=0, response_format={"type": "json_object"})

    assert replayed == response
    assert replay_wrapped.calls == []
    assert reader.stats() == {
        "enabled": True,
        "mode": "read",
        "provider": "openai",
        "model": "gpt-5-mini",
        "endpoint": "<provider-default>",
        "mem0_version": reader.mem0_version,
        "temperature": 0.0,
        "memory_extraction_response_protocol": MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
        "memory_extraction_max_model_len": MEMORY_EXTRACTION_MAX_MODEL_LEN,
        "memory_extraction_max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
        "memory_extraction_max_facts": MEMORY_EXTRACTION_MAX_FACTS,
        "memory_extraction_max_text_chars": MEMORY_EXTRACTION_MAX_TEXT_CHARS,
        "cache_dir": str(reader.cache_dir),
        "hits": 1,
        "misses": 0,
    }


def test_cached_embedder_isolated_by_normalized_endpoint(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    first = CachedEmbedder(
        RecordingEmbedder(),
        cache_dir=cache_dir,
        model="model",
        mode="write",
        endpoint="HTTPS://EXAMPLE.COM/v1/",
    )
    second = CachedEmbedder(
        RecordingEmbedder(),
        cache_dir=cache_dir,
        model="model",
        mode="write",
        endpoint="https://other.example/v1",
    )

    assert first.embed("same text", "add") == _vector_for("same text")
    assert second.embed("same text", "add") == _vector_for("same text")
    assert first.cache_dir != second.cache_dir
    assert first.endpoint == "https://example.com/v1"


def test_cached_memory_llm_read_mode_fails_on_missing_or_corrupt_entries_without_fallback(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    wrapped = RecordingMemoryLLM("must not be called")
    reader = CachedMemoryLLM(wrapped, cache_dir, "openai", "gpt-5-mini", "read")

    with pytest.raises(CachedMemoryLLMMissingError, match="--preembed-only"):
        reader.generate_response(messages, response_format={"type": "json_object"})
    assert wrapped.calls == []

    valid_response = _valid_extraction_response()
    writer = CachedMemoryLLM(
        RecordingMemoryLLM(valid_response), cache_dir, "openai", "gpt-5-mini", "write"
    )
    assert writer.generate_response(messages, response_format={"type": "json_object"}) == valid_response
    cache_file = next(writer.cache_dir.glob("*.json"))
    cache_file.write_text("not json", encoding="utf-8")

    with pytest.raises(CachedMemoryLLMMissingError, match="corrupt.*--preembed-only"):
        reader.generate_response(messages, response_format={"type": "json_object"})
    assert wrapped.calls == []


def test_cached_memory_llm_never_caches_invalid_extraction_response(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    wrapped = RecordingMemoryLLM('{"memory":[{"id":"0"}]')
    cached = CachedMemoryLLM(wrapped, cache_dir, "vllm", "model", "write")

    with pytest.raises(InvalidMemoryExtractionResponseError, match="malformed JSON"):
        cached.generate_response(messages, response_format={"type": "json_object"})

    assert len(wrapped.calls) == 1 + len(MEMORY_EXTRACTION_RETRY_TEMPERATURES)
    assert list(cached.cache_dir.glob("*.json")) == []
    dumps = sorted((cached.cache_dir / "invalid").glob("*.json"))
    assert len(dumps) == 1 + len(MEMORY_EXTRACTION_RETRY_TEMPERATURES)


def test_cached_memory_llm_retries_extraction_with_escalating_temperature(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    valid_response = _valid_extraction_response()
    wrapped = SequenceMemoryLLM(['{"memory":[{"id":"0"}]', valid_response])
    cached = CachedMemoryLLM(wrapped, cache_dir, "vllm", "model", "write")

    assert (
        cached.generate_response(messages, response_format={"type": "json_object"})
        == valid_response
    )
    assert len(wrapped.calls) == 2
    first_kwargs, retry_kwargs = wrapped.calls[0][2], wrapped.calls[1][2]
    assert "temperature" not in first_kwargs
    assert retry_kwargs["temperature"] == MEMORY_EXTRACTION_RETRY_TEMPERATURES[0]
    assert {
        key: value for key, value in retry_kwargs.items() if key != "temperature"
    } == first_kwargs

    dumps = list((cached.cache_dir / "invalid").glob("*.json"))
    assert len(dumps) == 1
    dump_payload = json.loads(dumps[0].read_text(encoding="utf-8"))
    assert dump_payload["attempt"] == 1
    assert dump_payload["response"] == '{"memory":[{"id":"0"}]'
    assert "malformed JSON" in dump_payload["error"]

    # The retried response is cached under the baseline digest and replays
    # without contacting the LLM again.
    reader_wrapped = RecordingMemoryLLM("must not be called")
    reader = CachedMemoryLLM(reader_wrapped, cache_dir, "vllm", "model", "read")
    assert (
        reader.generate_response(messages, response_format={"type": "json_object"})
        == valid_response
    )
    assert reader_wrapped.calls == []


def test_cached_memory_llm_raises_after_exhausting_extraction_retries(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    attempts = 1 + len(MEMORY_EXTRACTION_RETRY_TEMPERATURES)
    wrapped = SequenceMemoryLLM(['{"memory":[{"id":"0"}]'] * attempts)
    cached = CachedMemoryLLM(wrapped, cache_dir, "vllm", "model", "write")

    with pytest.raises(
        InvalidMemoryExtractionResponseError,
        match=f"failed validation on all {attempts} attempts",
    ):
        cached.generate_response(messages, response_format={"type": "json_object"})

    assert len(wrapped.calls) == attempts
    temperatures = [call[2].get("temperature") for call in wrapped.calls]
    assert temperatures == [None, *MEMORY_EXTRACTION_RETRY_TEMPERATURES]
    assert list(cached.cache_dir.glob("*.json")) == []
    dump_attempts = sorted(
        json.loads(dump.read_text(encoding="utf-8"))["attempt"]
        for dump in (cached.cache_dir / "invalid").glob("*.json")
    )
    assert dump_attempts == list(range(1, attempts + 1))


def test_cached_memory_llm_does_not_retry_non_extraction_calls(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    wrapped = RecordingMemoryLLM("plain response")
    cached = CachedMemoryLLM(wrapped, cache_dir, "vllm", "model", "write")

    assert cached.generate_response(messages) == "plain response"
    assert len(wrapped.calls) == 1
    assert not (cached.cache_dir / "invalid").exists()


def test_cached_memory_llm_write_mode_regenerates_invalid_cached_extraction(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    valid_response = _valid_extraction_response()
    seed = CachedMemoryLLM(
        RecordingMemoryLLM(valid_response), cache_dir, "vllm", "model", "write"
    )
    seed.generate_response(messages, response_format={"type": "json_object"})
    cache_file = next(seed.cache_dir.glob("*.json"))
    cache_file.write_text(
        '{"version":1,"response":"{\\"memory\\":[{\\"id\\":\\"0\\"}]}"}\n',
        encoding="utf-8",
    )

    wrapped = RecordingMemoryLLM(valid_response)
    replacement = CachedMemoryLLM(wrapped, cache_dir, "vllm", "model", "write")

    assert replacement.generate_response(
        messages, response_format={"type": "json_object"}
    ) == valid_response
    assert len(wrapped.calls) == 1


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"memory": [{"id": 1, "text": "fact", "attributed_to": "user"}]}, "must be a string"),
        (
            {"memory": [{"id": "0", "text": "fact", "attributed_to": "system"}]},
            "invalid attributed_to",
        ),
        (
            {
                "memory": [
                    {
                        "id": "0",
                        "text": "x" * (MEMORY_EXTRACTION_MAX_TEXT_CHARS + 1),
                        "attributed_to": "user",
                    }
                ]
            },
            "characters",
        ),
        (
            {
                "memory": [
                    {"id": str(index), "text": "fact", "attributed_to": "user"}
                    for index in range(MEMORY_EXTRACTION_MAX_FACTS + 1)
                ]
            },
            "maximum",
        ),
    ],
)
def test_memory_extraction_validation_rejects_protocol_violations(
    payload: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(InvalidMemoryExtractionResponseError, match=message):
        validate_memory_extraction_response(json.dumps(payload))


def test_memory_extraction_validation_normalizes_nonsequential_ids() -> None:
    payload = {
        "memory": [
            {"id": "1", "text": "first fact", "attributed_to": "user"},
            {"id": "e7b1c2d3", "text": "second fact", "attributed_to": "assistant"},
        ]
    }
    normalized = json.loads(validate_memory_extraction_response(json.dumps(payload)))
    assert [memory["id"] for memory in normalized["memory"]] == ["0", "1"]
    assert [memory["text"] for memory in normalized["memory"]] == ["first fact", "second fact"]


def test_memory_extraction_validation_preserves_sequential_response_verbatim() -> None:
    response = _valid_extraction_response()
    assert validate_memory_extraction_response(response) is response


def test_cached_memory_llm_write_mode_caches_normalized_extraction(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    raw_response = json.dumps(
        {"memory": [{"id": "1", "text": "User likes blue.", "attributed_to": "user"}]}
    )
    wrapped = RecordingMemoryLLM(raw_response)
    cache = CachedMemoryLLM(wrapped, cache_dir, "vllm", "model", "write")

    response = cache.generate_response(messages, response_format={"type": "json_object"})
    assert json.loads(response)["memory"][0]["id"] == "0"

    replay = CachedMemoryLLM(RecordingMemoryLLM("unused"), cache_dir, "vllm", "model", "read")
    cached = replay.generate_response(messages, response_format={"type": "json_object"})
    assert json.loads(cached)["memory"][0]["id"] == "0"


def test_cached_memory_llm_write_mode_replaces_corrupt_entries_atomically(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    first = CachedMemoryLLM(RecordingMemoryLLM("first"), cache_dir, "openai", "gpt-5-mini", "write")
    assert first.generate_response(messages) == "first"
    cache_file = next(first.cache_dir.glob("*.json"))
    cache_file.write_text("{}", encoding="utf-8")

    wrapped = RecordingMemoryLLM("second")
    replacement = CachedMemoryLLM(wrapped, cache_dir, "openai", "gpt-5-mini", "write")
    assert replacement.generate_response(messages) == "second"
    assert len(wrapped.calls) == 1
    assert list(cache_file.parent.glob("*.tmp")) == []

    reader = CachedMemoryLLM(RecordingMemoryLLM("unused"), cache_dir, "openai", "gpt-5-mini", "read")
    assert reader.generate_response(messages) == "second"


def test_cached_memory_llm_ignores_current_date_but_preserves_observation_date(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages_day_one = [
        {
            "role": "user",
            "content": (
                "## New Messages\nAlice met Bob on 2024-01-02.\n\n"
                "## Observation Date\n2026-07-10\n\n"
                "## Current Date\n2026-07-10"
            ),
        }
    ]
    messages_day_two = [
        {
            "role": "user",
            "content": (
                "## New Messages\nAlice met Bob on 2024-01-02.\n\n"
                "## Observation Date\n2026-07-10\n\n"
                "## Current Date\n2026-07-11"
            ),
        }
    ]
    writer = CachedMemoryLLM(
        RecordingMemoryLLM("cached extraction"),
        cache_dir,
        "openai",
        "gpt-5-mini",
        "write",
    )
    assert writer.generate_response(messages_day_one) == "cached extraction"

    wrapped = RecordingMemoryLLM("must not be called")
    reader = CachedMemoryLLM(wrapped, cache_dir, "openai", "gpt-5-mini", "read")
    assert reader.generate_response(messages_day_two) == "cached extraction"
    assert wrapped.calls == []

    changed_observation_date = [
        {
            "role": "user",
            "content": messages_day_two[0]["content"].replace(
                "## Observation Date\n2026-07-10",
                "## Observation Date\n2026-07-11",
            ),
        }
    ]
    with pytest.raises(CachedMemoryLLMMissingError, match="--preembed-only"):
        reader.generate_response(changed_observation_date)

    changed_conversation = [
        {
            "role": "user",
            "content": messages_day_two[0]["content"].replace("Alice met Bob", "Alice met Carol"),
        }
    ]
    with pytest.raises(CachedMemoryLLMMissingError, match="--preembed-only"):
        reader.generate_response(changed_conversation)


def test_cached_memory_llm_isolated_by_normalized_endpoint(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    first_wrapped = RecordingMemoryLLM("first")
    second_wrapped = RecordingMemoryLLM("second")

    first = CachedMemoryLLM(
        first_wrapped,
        cache_dir,
        "openai",
        "gpt-5-mini",
        "write",
        endpoint="HTTPS://EXAMPLE.COM/v1/",
    )
    second = CachedMemoryLLM(
        second_wrapped,
        cache_dir,
        "openai",
        "gpt-5-mini",
        "write",
        endpoint="https://other.example/v1",
    )

    assert first.generate_response(messages) == "first"
    assert second.generate_response(messages) == "second"
    assert first.cache_dir != second.cache_dir
    assert first.endpoint == "https://example.com/v1"
    assert len(first_wrapped.calls) == 1
    assert len(second_wrapped.calls) == 1


def test_cached_memory_llm_temperature_zero_keeps_baseline_cache_keys(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]

    baseline = CachedMemoryLLM(RecordingMemoryLLM("cached"), cache_dir, "openai", "gpt-5-mini", "write")
    assert baseline.generate_response(messages) == "cached"

    explicit_zero = CachedMemoryLLM(
        RecordingMemoryLLM("unused"),
        cache_dir,
        "openai",
        "gpt-5-mini",
        "read",
        temperature=0.0,
    )
    assert explicit_zero.generate_response(messages) == "cached"
    assert explicit_zero.hits == 1


def test_cached_memory_llm_nonzero_temperature_gets_distinct_cache_keys(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]

    zero_writer = CachedMemoryLLM(RecordingMemoryLLM("zero"), cache_dir, "openai", "gpt-5-mini", "write")
    assert zero_writer.generate_response(messages) == "zero"

    warm_reader = CachedMemoryLLM(
        RecordingMemoryLLM("unused"),
        cache_dir,
        "openai",
        "gpt-5-mini",
        "read",
        temperature=0.1,
    )
    with pytest.raises(CachedMemoryLLMMissingError):
        warm_reader.generate_response(messages)
    assert warm_reader.stats()["temperature"] == 0.1
