from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from ..cache_identity import atomic_write_json, endpoint_cache_key, safe_path_part
from ..cache_identity import normalize_endpoint as normalize_llm_endpoint
from ..embedding.cache import CacheMode
from ..protocol import (
    MEMORY_EXTRACTION_MAX_FACTS,
    MEMORY_EXTRACTION_MAX_MODEL_LEN,
    MEMORY_EXTRACTION_MAX_TEXT_CHARS,
    MEMORY_EXTRACTION_MAX_TOKENS,
    MEMORY_EXTRACTION_RESPONSE_PROTOCOL,
    MEMORY_EXTRACTION_RETRY_TEMPERATURES,
)
from .memory_llm_protocol import (
    InvalidMemoryExtractionResponseError,
    prepare_memory_extraction_kwargs,
    validate_memory_extraction_response,
)


_MEM0_DYNAMIC_DATE_SECTION = re.compile(
    r"(?m)^(## Current Date\n)\d{4}-\d{2}-\d{2}$"
)


class CachedMemoryLLMMissingError(RuntimeError):
    pass


class CachedMemoryLLM:
    """Disk-backed response cache for a Mem0 LLM provider."""

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
        self.endpoint = normalize_llm_endpoint(endpoint)
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
        effective_kwargs, validate_extraction = prepare_memory_extraction_kwargs(kwargs)
        path = self._cache_path(messages, args, effective_kwargs)
        if path.exists():
            try:
                response = self._read_response(path)
                if validate_extraction:
                    response = validate_memory_extraction_response(response)
            except Exception as exc:
                self.misses += 1
                if self.mode == "read":
                    raise self._missing_error(path, "Cached Mem0 LLM response is corrupt") from exc
                path.unlink(missing_ok=True)
            else:
                self.hits += 1
                return response
        else:
            self.misses += 1

        if self.mode == "read":
            raise self._missing_error(path, "Missing cached Mem0 LLM response")

        if validate_extraction:
            response = self._generate_validated_extraction(
                path, messages, args, effective_kwargs
            )
        else:
            response = self._wrapped.generate_response(messages, *args, **effective_kwargs)
        self._write_response(path, response)
        return response

    def _generate_validated_extraction(
        self,
        path: Path,
        messages: Any,
        args: tuple[Any, ...],
        effective_kwargs: dict[str, Any],
    ) -> str:
        # Attempt 1 keeps the exact baseline request; retries only exist on
        # failure paths and escalate temperature so greedy decoding cannot
        # deterministically reproduce a degenerate response. The cache path is
        # derived from the baseline kwargs, so retries never change identity.
        attempt_kwargs = [effective_kwargs] + [
            {**effective_kwargs, "temperature": temperature}
            for temperature in MEMORY_EXTRACTION_RETRY_TEMPERATURES
        ]
        dump_paths: list[Path] = []
        last_error: InvalidMemoryExtractionResponseError | None = None
        for attempt, kwargs in enumerate(attempt_kwargs, start=1):
            response = self._wrapped.generate_response(messages, *args, **kwargs)
            try:
                return validate_memory_extraction_response(response)
            except InvalidMemoryExtractionResponseError as exc:
                last_error = exc
                dump_paths.append(
                    self._dump_invalid_response(path, response, attempt, kwargs, exc)
                )
                logger.warning(
                    "Mem0 extraction attempt {}/{} failed validation "
                    "model={} temperature={}: {}",
                    attempt,
                    len(attempt_kwargs),
                    self.model,
                    kwargs.get("temperature", self.temperature),
                    exc,
                )
        assert last_error is not None
        raise InvalidMemoryExtractionResponseError(
            f"{last_error} Mem0 extraction failed validation on all "
            f"{len(attempt_kwargs)} attempts; raw responses saved to: "
            + ", ".join(str(dump_path) for dump_path in dump_paths)
        ) from last_error

    def _dump_invalid_response(
        self,
        path: Path,
        response: Any,
        attempt: int,
        kwargs: dict[str, Any],
        error: Exception,
    ) -> Path:
        dump_path = path.parent / "invalid" / f"{path.stem}.attempt{attempt}.json"
        atomic_write_json(
            dump_path,
            {
                "version": 1,
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

    def _cache_path(self, messages: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path:
        request = {
            "provider": self.provider,
            "model": self.model,
            "endpoint": self.endpoint,
            "mem0_version": self.mem0_version,
            "messages": _normalize_mem0_prompt_dates(messages),
            "args": args,
            "kwargs": kwargs,
        }
        # 0.0 is the historical baseline identity: entries cached before
        # temperature entered the key stay valid, while any other temperature
        # gets distinct digests instead of silently replaying 0.0 responses.
        if self.temperature != 0.0:
            request["temperature"] = self.temperature
        try:
            canonical = json.dumps(
                request,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("Mem0 LLM cache inputs must be JSON-serializable.") from exc
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    @staticmethod
    def _read_response(path: Path) -> Any:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1 or "response" not in payload:
            raise ValueError("Unsupported Mem0 LLM cache entry format.")
        return payload["response"]

    @staticmethod
    def _write_response(path: Path, response: Any) -> None:
        atomic_write_json(path, {"version": 1, "response": response})

    def _missing_error(self, path: Path, reason: str) -> CachedMemoryLLMMissingError:
        return CachedMemoryLLMMissingError(
            f"{reason} for provider={self.provider!r} model={self.model!r} "
            f"endpoint={self.endpoint!r} mem0ai={self.mem0_version!r} at {path}. "
            "Rerun locomo-jasper-bench --preembed-only with the same dataset, memory LLM model, "
            "and memory LLM cache dir before running the benchmark."
        )


def _normalize_mem0_prompt_dates(value: Any) -> Any:
    if isinstance(value, str):
        return _MEM0_DYNAMIC_DATE_SECTION.sub(r"\1<preembed-date>", value)
    if isinstance(value, list):
        return [_normalize_mem0_prompt_dates(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_mem0_prompt_dates(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_mem0_prompt_dates(item) for key, item in value.items()}
    return value
