from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from locomo_jasper_bench.embedding.cache import CachedEmbedder, CachedEmbeddingMissingError
from locomo_jasper_bench.retrieval.memory_llm_cache import CachedMemoryLLM, CachedMemoryLLMMissingError


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


def _vector_for(value: Any) -> list[float]:
    text = str(value)
    return [float(len(text)), float(sum(text.encode("utf-8")))]


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


def test_cached_memory_llm_replays_canonical_requests_and_delegates_attributes(tmp_path: Path) -> None:
    cache_dir = tmp_path / "memory-llm"
    messages = [{"role": "user", "content": "Remember blue."}]
    response = {"content": '{"memory":[{"text":"blue"}]}', "tool_calls": []}
    wrapped = RecordingMemoryLLM(response)
    writer = CachedMemoryLLM(wrapped, cache_dir, "openai", "gpt-5-mini", "write")

    assert writer.generate_response(messages, response_format={"type": "json_object"}, temperature=0) == response
    assert len(wrapped.calls) == 1
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

    writer = CachedMemoryLLM(RecordingMemoryLLM("cached"), cache_dir, "openai", "gpt-5-mini", "write")
    assert writer.generate_response(messages, response_format={"type": "json_object"}) == "cached"
    cache_file = next(writer.cache_dir.glob("*.json"))
    cache_file.write_text("not json", encoding="utf-8")

    with pytest.raises(CachedMemoryLLMMissingError, match="corrupt.*--preembed-only"):
        reader.generate_response(messages, response_format={"type": "json_object"})
    assert wrapped.calls == []


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
