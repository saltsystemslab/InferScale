from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.cache_keys import endpoint_cache_key, normalize_endpoint, safe_path_part
from benchmarks.common.embedding_cache import CacheMode
from benchmarks.memory.mem0.llm_protocol import (
    InvalidMemoryExtractionResponseError,
    prepare_memory_extraction_kwargs,
    validate_memory_extraction_response,
)
from benchmarks.memory.protocol import (
    MEMORY_EXTRACTION_MAX_FACTS,
    MEMORY_EXTRACTION_MAX_MODEL_LEN,
    MEMORY_EXTRACTION_MAX_TEXT_CHARS,
    MEMORY_EXTRACTION_MAX_TOKENS,
    MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
    MEMORY_EXTRACTION_RETRY_TEMPERATURES,
    MEMORY_EXTRACTION_RETRY_TOP_P,
)

CURRENT_DATE_SECTION = re.compile(r"(?m)^(## Current Date\n)\d{4}-\d{2}-\d{2}$")
CURRENT_DATE_PLACEHOLDER = r"\1<preembed-date>"
UNKEYED_TEMPERATURE = 0.0
CACHE_ENTRY_VERSION = 1


class CachedMemoryLLMMissingError(RuntimeError):
    pass


class CachedMemoryLLM:
    def __init__(
        self,
        wrapped: Any,
        cache_dir: str | Path,
        provider: str,
        model: str,
        mode: CacheMode,
        endpoint: str | None = None,
        mem0_version: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        if mode not in {"read", "write"}:
            raise ValueError(f"Unsupported memory LLM cache mode: {mode}")
        self._wrapped = wrapped
        self.provider = provider
        self.model = model
        self.endpoint = normalize_endpoint(endpoint)
        self.mem0_version = mem0_version or importlib.metadata.version("mem0ai")
        self.temperature = float(temperature)
        self.mode = mode
        self.cache_dir = (
            Path(cache_dir)
            / safe_path_part(provider)
            / safe_path_part(model)
            / f"mem0-{safe_path_part(self.mem0_version)}"
            / safe_path_part(MEMORY_EXTRACTION_RESPONSE_PROTOCOL)
            / f"context-{MEMORY_EXTRACTION_MAX_MODEL_LEN}"
            / endpoint_cache_key(self.endpoint)
        )
        self.hits = 0
        self.misses = 0
        if self.mode == "write":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def generate_response(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        effective_kwargs, is_extraction = prepare_memory_extraction_kwargs(kwargs)
        entry_path = self._cache_path(messages, args, effective_kwargs)
        cached_response = self._load_cached(entry_path, is_extraction)
        if cached_response is not None:
            self.hits += 1
            return cached_response
        self.misses += 1
        if self.mode == "read":
            raise self._missing_error(entry_path, "Missing cached Mem0 LLM response")
        if is_extraction:
            response = self._generate_validated_extraction(
                entry_path, messages, args, effective_kwargs
            )
        else:
            response = self._wrapped.generate_response(messages, *args, **effective_kwargs)
        atomic_write_json(entry_path, {"version": CACHE_ENTRY_VERSION, "response": response})
        return response

    def stats(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "mem0_version": self.mem0_version,
            "temperature": self.temperature,
            "memory_extraction_response_protocol": MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
            "memory_extraction_max_model_len": MEMORY_EXTRACTION_MAX_MODEL_LEN,
            "memory_extraction_max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
            "memory_extraction_max_facts": MEMORY_EXTRACTION_MAX_FACTS,
            "memory_extraction_max_text_chars": MEMORY_EXTRACTION_MAX_TEXT_CHARS,
            "cache_dir": str(self.cache_dir),
            "hits": self.hits,
            "misses": self.misses,
        }

    def _load_cached(self, entry_path: Path, is_extraction: bool) -> Any | None:
        if not entry_path.exists():
            return None
        try:
            response = _read_entry(entry_path)
            return validate_memory_extraction_response(response) if is_extraction else response
        except Exception as exc:
            if self.mode == "read":
                self.misses += 1
                raise self._missing_error(
                    entry_path, "Cached Mem0 LLM response is corrupt"
                ) from exc
            entry_path.unlink(missing_ok=True)
            return None

    def _generate_validated_extraction(
        self,
        entry_path: Path,
        messages: Any,
        args: tuple[Any, ...],
        baseline_kwargs: dict[str, Any],
    ) -> str:
        retry_kwargs = [
            {**baseline_kwargs, "temperature": temperature, "top_p": MEMORY_EXTRACTION_RETRY_TOP_P}
            for temperature in MEMORY_EXTRACTION_RETRY_TEMPERATURES
        ]
        kwargs_per_attempt = [baseline_kwargs, *retry_kwargs]
        attempt_count = len(kwargs_per_attempt)
        dump_paths: list[Path] = []
        last_error: InvalidMemoryExtractionResponseError | None = None
        for attempt, attempt_kwargs in enumerate(kwargs_per_attempt, start=1):
            response = self._wrapped.generate_response(messages, *args, **attempt_kwargs)
            try:
                return validate_memory_extraction_response(response)
            except InvalidMemoryExtractionResponseError as exc:
                last_error = exc
                dump_paths.append(
                    self._dump_invalid_response(entry_path, response, attempt, attempt_kwargs, exc)
                )
                logger.warning(
                    "Mem0 extraction attempt {}/{} failed validation model={} temperature={}: {}",
                    attempt,
                    attempt_count,
                    self.model,
                    attempt_kwargs.get("temperature", self.temperature),
                    exc,
                )
        assert last_error is not None
        raise InvalidMemoryExtractionResponseError(
            f"{last_error} Mem0 extraction failed validation on all {attempt_count} attempts; "
            "raw responses saved to: " + ", ".join(str(dump_path) for dump_path in dump_paths)
        ) from last_error

    def _dump_invalid_response(
        self,
        entry_path: Path,
        response: Any,
        attempt: int,
        attempt_kwargs: dict[str, Any],
        error: Exception,
    ) -> Path:
        dump_path = entry_path.parent / "invalid" / f"{entry_path.stem}.attempt{attempt}.json"
        atomic_write_json(
            dump_path,
            {
                "version": CACHE_ENTRY_VERSION,
                "provider": self.provider,
                "model": self.model,
                "endpoint": self.endpoint,
                "mem0_version": self.mem0_version,
                "attempt": attempt,
                "temperature": attempt_kwargs.get("temperature", self.temperature),
                "error": str(error),
                "response": response if isinstance(response, str) else repr(response),
            },
            indent=2,
        )
        return dump_path

    def _cache_path(self, messages: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path:
        request = {
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "mem0_version": self.mem0_version,
            "messages": _replayable_on_another_day(messages),
            "args": args,
            "kwargs": kwargs,
        }
        if self.temperature != UNKEYED_TEMPERATURE:
            request["temperature"] = self.temperature
        try:
            canonical_request = json.dumps(
                request, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("Mem0 LLM cache inputs must be JSON-serializable.") from exc
        digest = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _missing_error(self, entry_path: Path, reason: str) -> CachedMemoryLLMMissingError:
        return CachedMemoryLLMMissingError(
            f"{reason} for provider={self.provider!r} model={self.model!r} "
            f"endpoint={self.endpoint!r} mem0ai={self.mem0_version!r} at {entry_path}. "
            "Rerun the extract-facts stage with the same dataset, memory LLM model, "
            "and memory LLM cache dir before running the benchmark."
        )


def _read_entry(entry_path: Path) -> Any:
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    if (
        not isinstance(entry, dict)
        or entry.get("version") != CACHE_ENTRY_VERSION
        or "response" not in entry
    ):
        raise ValueError("Unsupported Mem0 LLM cache entry format.")
    return entry["response"]


def _replayable_on_another_day(prompt: Any) -> Any:
    if isinstance(prompt, str):
        return CURRENT_DATE_SECTION.sub(CURRENT_DATE_PLACEHOLDER, prompt)
    if isinstance(prompt, list):
        return [_replayable_on_another_day(part) for part in prompt]
    if isinstance(prompt, tuple):
        return tuple(_replayable_on_another_day(part) for part in prompt)
    if isinstance(prompt, dict):
        return {key: _replayable_on_another_day(part) for key, part in prompt.items()}
    return prompt
