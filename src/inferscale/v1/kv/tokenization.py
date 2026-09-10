from __future__ import annotations

from typing import Any


def encode_text_no_special(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Tokenizer has no encode method.")
    return list(encode(text, add_special_tokens=False))
