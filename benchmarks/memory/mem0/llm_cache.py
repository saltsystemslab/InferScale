"""Disk cache of Mem0 extraction responses, with validated retries on the write path.

Extraction runs once per model; later runs replay the cached responses so the
LLM never influences timed measurements. Entries are keyed by the request
Mem0 made, minus the current date it injects into every prompt.
"""

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

# Mem0 writes today's date into the extraction prompt. The observation date
# section stays part of the key; the current date is replaced so a prompt
# replayed on another day still hits the cache.
CURRENT_DATE_SECTION = re.compile(r"(?m)^(## Current Date\n)\d{4}-\d{2}-\d{2}$")
CURRENT_DATE_PLACEHOLDER = r"\1<preembed-date>"
CACHE_ENTRY_VERSION = 1


class CachedMemoryLLMMissingError(RuntimeError):
    """Read mode found no usable cached response for a request."""


class CachedMemoryLLM:
    """Wrap a Mem0 LLM provider so identical extraction requests replay from disk."""

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
        # Mem0 reads provider attributes such as ``config`` from its LLM.
        return getattr(self._wrapped, name)

    def generate_response(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        effective_kwargs, validate_extraction = prepare_memory_extraction_kwargs(kwargs)
        path = self._cache_path(messages, args, effective_kwargs)
        cached = self._load_cached(path, validate_extraction)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        if self.mode == "read":
            raise self._missing_error(path, "Missing cached Mem0 LLM response")
        if validate_extraction:
            response = self._generate_validated_extraction(path, messages, args, effective_kwargs)
        else:
            response = self._wrapped.generate_response(messages, *args, **effective_kwargs)
        atomic_write_json(path, {"version": CACHE_ENTRY_VERSION, "response": response})
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

    def _load_cached(self, path: Path, validate_extraction: bool) -> Any | None:
        """The cached response, or None when absent; corrupt entries are replaced in write mode."""
        if not path.exists():
            return None
        try:
            response = _read_entry(path)
            return (
                validate_memory_extraction_response(response) if validate_extraction else response
            )
        except Exception as exc:
            if self.mode == "read":
                self.misses += 1
                raise self._missing_error(path, "Cached Mem0 LLM response is corrupt") from exc
            path.unlink(missing_ok=True)
            return None

    def _generate_validated_extraction(
        self,
        path: Path,
        messages: Any,
        args: tuple[Any, ...],
        effective_kwargs: dict[str, Any],
    ) -> str:
        """Call the LLM until it returns a valid extraction, escalating the temperature.

        The first attempt uses the baseline request. Retries raise the
        temperature and widen top_p (Mem0 otherwise sends its 0.1 default on
        every request) so greedy decoding cannot reproduce the same degenerate
        response. The cache path comes from the baseline kwargs, so retries never
        change the entry's identity.
        """
        attempts = [effective_kwargs] + [
            {**effective_kwargs, "temperature": temperature, "top_p": MEMORY_EXTRACTION_RETRY_TOP_P}
            for temperature in MEMORY_EXTRACTION_RETRY_TEMPERATURES
        ]
        dump_paths: list[Path] = []
        last_error: InvalidMemoryExtractionResponseError | None = None
        for attempt, kwargs in enumerate(attempts, start=1):
            response = self._wrapped.generate_response(messages, *args, **kwargs)
            try:
                return validate_memory_extraction_response(response)
            except InvalidMemoryExtractionResponseError as exc:
                last_error = exc
                dump_paths.append(self._dump_invalid_response(path, response, attempt, kwargs, exc))
                logger.warning(
                    "Mem0 extraction attempt {}/{} failed validation model={} temperature={}: {}",
                    attempt,
                    len(attempts),
                    self.model,
                    kwargs.get("temperature", self.temperature),
                    exc,
                )
        assert last_error is not None
        raise InvalidMemoryExtractionResponseError(
            f"{last_error} Mem0 extraction failed validation on all {len(attempts)} attempts; "
            "raw responses saved to: " + ", ".join(str(dump_path) for dump_path in dump_paths)
        ) from last_error

    def _dump_invalid_response(
        self, path: Path, response: Any, attempt: int, kwargs: dict[str, Any], error: Exception
    ) -> Path:
        dump_path = path.parent / "invalid" / f"{path.stem}.attempt{attempt}.json"
        atomic_write_json(
            dump_path,
            {
                "version": CACHE_ENTRY_VERSION,
                "provider": self.provider,
                "model": self.model,
                "endpoint": self.endpoint,
                "mem0_version": self.mem0_version,
                "attempt": attempt,
                "temperature": kwargs.get("temperature", self.temperature),
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
            "messages": _without_current_date(messages),
            "args": args,
            "kwargs": kwargs,
        }
        # Cache format rule: the key carries the temperature only when it is
        # not 0.0, so every non-zero temperature gets its own entries.
        if self.temperature != 0.0:
            request["temperature"] = self.temperature
        try:
            canonical = json.dumps(
                request, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("Mem0 LLM cache inputs must be JSON-serializable.") from exc
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _missing_error(self, path: Path, reason: str) -> CachedMemoryLLMMissingError:
        return CachedMemoryLLMMissingError(
            f"{reason} for provider={self.provider!r} model={self.model!r} "
            f"endpoint={self.endpoint!r} mem0ai={self.mem0_version!r} at {path}. "
            "Rerun the extract-facts stage with the same dataset, memory LLM model, "
            "and memory LLM cache dir before running the benchmark."
        )


def _read_entry(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("version") != CACHE_ENTRY_VERSION
        or "response" not in payload
    ):
        raise ValueError("Unsupported Mem0 LLM cache entry format.")
    return payload["response"]


def _without_current_date(value: Any) -> Any:
    if isinstance(value, str):
        return CURRENT_DATE_SECTION.sub(CURRENT_DATE_PLACEHOLDER, value)
    if isinstance(value, list):
        return [_without_current_date(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_current_date(item) for item in value)
    if isinstance(value, dict):
        return {key: _without_current_date(item) for key, item in value.items()}
    return value
