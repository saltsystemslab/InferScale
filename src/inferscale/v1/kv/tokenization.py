"""Tokenization of raw text without special tokens, shared by encoder and prompt code."""

from __future__ import annotations

from typing import Any


def encode_text_no_special(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))
