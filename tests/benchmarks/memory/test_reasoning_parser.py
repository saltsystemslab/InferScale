from __future__ import annotations

from memory_config import memory_runtime


def test_configured_qwen3_alias_resolves_the_reasoning_parser() -> None:
    assert memory_runtime().reasoning_parser("qwen3-14b") == "qwen3"


def test_alias_keeps_the_parser_when_the_checkpoint_is_overridden() -> None:
    runtime = memory_runtime(models={"qwen3-14b": "/workspace/models/my-14b"})

    assert runtime.reasoning_parser("qwen3-14b") == "qwen3"
    assert runtime.reasoning_parser("/workspace/models/my-14b") == "qwen3"


def test_raw_configured_qwen3_id_resolves_the_parser() -> None:
    assert memory_runtime().reasoning_parser("Qwen/Qwen3-14B") == "qwen3"


def test_non_reasoning_models_resolve_no_parser() -> None:
    runtime = memory_runtime()

    assert runtime.reasoning_parser("llama") is None
    assert runtime.reasoning_parser("meta-llama/Llama-3.1-8B-Instruct") is None
    assert runtime.reasoning_parser("some/other-model") is None


def test_reasoning_parser_aliases_are_explicit_runtime_configuration() -> None:
    runtime = memory_runtime(
        models={"qwen3": "Qwen/Qwen3-14B"},
        reasoning_parsers={"qwen3": "qwen3"},
    )

    assert runtime.reasoning_parser("qwen3") == "qwen3"
    assert runtime.reasoning_parser("Qwen/Qwen3-14B") == "qwen3"
