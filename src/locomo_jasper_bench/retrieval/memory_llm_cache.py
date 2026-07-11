from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

from ..cache_identity import atomic_write_json, endpoint_cache_key, safe_path_part
from ..cache_identity import normalize_endpoint as normalize_llm_endpoint
from ..embedding.cache import CacheMode


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
            / endpoint_cache_key(self.endpoint)
        )
        self.hits = 0
        self.misses = 0
        if self.mode == "write":
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def generate_response(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        path = self._cache_path(messages, args, kwargs)
        if path.exists():
            try:
                response = self._read_response(path)
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

        response = self._wrapped.generate_response(messages, *args, **kwargs)
        self._write_response(path, response)
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
