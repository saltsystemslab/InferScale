from __future__ import annotations

import pytest

from locomo_jasper_bench.config import resolve_reasoning_parser


MODEL_ENV_VARS = (
    "LOCOMO_MODEL_LLAMA",
    "MODEL_LLAMA",
    "LOCOMO_MODEL_QWEN3_14B",
    "MODEL_QWEN3_14B",
)


def _clear_model_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MODEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_qwen3_aliases_resolve_the_reasoning_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_environment(monkeypatch)

    assert resolve_reasoning_parser("qwen3-14b") == "qwen3"
    assert resolve_reasoning_parser("qwen3") == "qwen3"
    assert resolve_reasoning_parser("QWEN3-14B") == "qwen3"


def test_alias_keeps_the_parser_when_the_checkpoint_is_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_model_environment(monkeypatch)
    monkeypatch.setenv("MODEL_QWEN3_14B", "/workspace/models/my-14b")

    assert resolve_reasoning_parser("qwen3-14b") == "qwen3"
    assert resolve_reasoning_parser("/workspace/models/my-14b") == "qwen3"


def test_raw_configured_qwen3_id_resolves_the_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_environment(monkeypatch)

    assert resolve_reasoning_parser("Qwen/Qwen3-14B") == "qwen3"


def test_non_reasoning_models_resolve_no_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_environment(monkeypatch)

    assert resolve_reasoning_parser("llama") is None
    assert resolve_reasoning_parser("meta-llama/Llama-3.1-8B-Instruct") is None
    assert resolve_reasoning_parser("some/other-model") is None
