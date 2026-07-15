from __future__ import annotations

import os

import pytest

from locomo_jasper_bench.throughput.kv_condition import (
    _TOKENIZER_PARITY_PROBE_TEXT,
    _require_tokenizer_parity,
)
from locomo_jasper_bench.kv.tokenization import encode_text_no_special


class StubTokenizer:
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False
        return [byte + self.offset for byte in text.encode("utf-8")]


def test_tokenizer_parity_passes_for_identical_tokenizers() -> None:
    probe_ids = encode_text_no_special(StubTokenizer(), _TOKENIZER_PARITY_PROBE_TEXT)
    _require_tokenizer_parity(probe_ids, StubTokenizer())


def test_tokenizer_parity_raises_for_diverging_tokenizers() -> None:
    probe_ids = encode_text_no_special(StubTokenizer(), _TOKENIZER_PARITY_PROBE_TEXT)
    with pytest.raises(RuntimeError, match="tokenizers disagree"):
        _require_tokenizer_parity(probe_ids, StubTokenizer(offset=1))


@pytest.mark.parametrize(
    "model",
    [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-14B",
    ],
)
def test_encoder_loader_matches_engine_loader(model: str) -> None:
    """Pod/cluster integration check: the encoder's tokenizer loader must
    produce the same ids as the engine's for every benchmarked model."""
    if not os.environ.get("RUN_TOKENIZER_PARITY_INTEGRATION"):
        pytest.skip("Set RUN_TOKENIZER_PARITY_INTEGRATION=1 on a machine with vllm and HF access.")
    pytest.importorskip("vllm")
    from vllm.transformers_utils.tokenizer import get_tokenizer

    tokenizer = get_tokenizer(model)
    probe_ids = encode_text_no_special(tokenizer, _TOKENIZER_PARITY_PROBE_TEXT)
    _require_tokenizer_parity(probe_ids, tokenizer)
