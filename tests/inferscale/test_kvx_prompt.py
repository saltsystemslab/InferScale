from __future__ import annotations

from typing import Any

from inferscale.v1.kv.prompt import apply_chat_template_non_thinking


def test_chat_template_is_retried_without_the_thinking_switch_it_rejects() -> None:
    calls: list[dict[str, Any]] = []

    class NoThinkingSwitchTokenizer:
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            calls.append(kwargs)
            if "enable_thinking" in kwargs:
                raise TypeError("got an unexpected keyword argument 'enable_thinking'")
            return "rendered"

    rendered = apply_chat_template_non_thinking(
        NoThinkingSwitchTokenizer(), [{"role": "user", "content": "Q"}], tokenize=False
    )

    assert rendered == "rendered"
    assert calls == [{"enable_thinking": False, "tokenize": False}, {"tokenize": False}]
