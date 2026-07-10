from __future__ import annotations

from typing import Any

from locomo_jasper_bench.throughput.synthetic import (
    build_memory_prompt,
    build_requests,
    build_synthetic_memory,
)


class CharacterTokenizer:
    bos_token_id = 1

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        values = [ord(character) + 10 for character in text]
        return ([self.bos_token_id] + values) if add_special_tokens else values

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return "".join(chr(token_id - 10) for token_id in token_ids if token_id != self.bos_token_id)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str | list[int]:
        text = "<chat>" + "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        if not tokenize:
            return text
        return [self.bos_token_id, *self.encode(text)]


def test_synthetic_memory_is_exact_and_deterministic() -> None:
    tokenizer = CharacterTokenizer()
    first = build_synthetic_memory(
        tokenizer,
        user_index=7,
        memory_tokens=512,
        seed=42,
        entry_tokens=64,
    )
    second = build_synthetic_memory(
        tokenizer,
        user_index=7,
        memory_tokens=512,
        seed=42,
        entry_tokens=64,
    )

    assert len(first.memory_token_ids) == 512
    assert first == second
    assert first.user_id == "user_0007"
    assert first.entries


def test_requests_do_not_change_with_user_count() -> None:
    small = build_requests(num_users=2, requests_per_user=2, seed=42)
    large = build_requests(num_users=5, requests_per_user=2, seed=42)

    assert small == large[: len(small)]


def test_memory_prompt_keeps_the_exact_memory_prefix() -> None:
    tokenizer = CharacterTokenizer()
    memory = build_synthetic_memory(tokenizer, user_index=0, memory_tokens=512, seed=42)
    prompt = build_memory_prompt(tokenizer, memory.memory_token_ids, "What should I read?")

    assert prompt[:512] == list(memory.memory_token_ids)
    assert len(prompt) > 512
