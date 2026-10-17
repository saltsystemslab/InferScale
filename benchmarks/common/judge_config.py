"""The ``judge`` section of a benchmark JSON file, layered over the runtime's judge server."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from benchmarks.common.config import REDACTED, RuntimeConfig
from benchmarks.common.json_fields import ConfigError, optional, reject_unknown_keys

DEFAULT_JUDGE_PROVIDER = "vllm"
DEFAULT_JUDGE_MAX_TOKENS = 4
JUDGE_PROVIDERS = ("vllm", "none")
JUDGE_KEYS = ("provider", "model", "base_url", "max_tokens", "with_evidence")


@dataclass(slots=True)
class JudgeConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    max_tokens: int
    with_evidence: bool = False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": REDACTED if self.api_key else None,
            "max_tokens": self.max_tokens,
            "with_evidence": self.with_evidence,
        }


def judge_config(section: Mapping[str, Any], runtime: RuntimeConfig, *, where: str) -> JudgeConfig:
    """Parse a judge section; the model and endpoint default to the runtime's judge server."""
    reject_unknown_keys(section, JUDGE_KEYS, where)
    provider = optional(section, "provider", str, DEFAULT_JUDGE_PROVIDER, where)
    if provider not in JUDGE_PROVIDERS:
        raise ConfigError(f"{where}.provider must be vllm or none, got {provider!r}.")
    max_tokens = optional(section, "max_tokens", int, DEFAULT_JUDGE_MAX_TOKENS, where)
    if max_tokens < 1:
        raise ConfigError(f"{where}.max_tokens must be >= 1.")
    return JudgeConfig(
        provider=provider,
        model=optional(section, "model", str, runtime.judge_server.model, where),
        base_url=optional(section, "base_url", str, runtime.judge_server.base_url, where),
        api_key=runtime.judge_api_key(),
        max_tokens=max_tokens,
        with_evidence=optional(section, "with_evidence", bool, False, where),
    )
