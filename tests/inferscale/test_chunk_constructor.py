from __future__ import annotations

import pytest

from inferscale.v1 import Chunk


def test_chunk_accepts_text_without_explicit_tokens_or_payload() -> None:
    chunk = Chunk("first", "First chunk.")

    assert chunk.id == "first"
    assert chunk.text == "First chunk."
    assert chunk.token_ids is None
    assert chunk.payload is None
    assert not hasattr(chunk, "context_token_ids")
    assert not hasattr(chunk, "context_ids")


def test_chunk_accepts_positional_tokens_and_keyword_payload() -> None:
    token_ids = (1, 2, 3)
    payload = {"source": "document"}
    chunk = Chunk("first", "First chunk.", token_ids, payload=payload)

    assert chunk.token_ids == token_ids
    assert chunk.payload == payload


@pytest.mark.parametrize("kwargs", [
    {"context_token_ids": [1, 2]},
    {"context_ids": ("previous",)},
])
def test_chunk_rejects_removed_context_keywords(kwargs: dict) -> None:
    with pytest.raises(TypeError, match=next(iter(kwargs))):
        Chunk("first", "First chunk.", **kwargs)


@pytest.mark.parametrize("fourth_argument", [[1, 2], {"source": "document"}])
def test_chunk_rejects_positional_context_or_payload(fourth_argument) -> None:
    with pytest.raises(TypeError):
        Chunk("first", "First chunk.", [3, 4], fourth_argument)
