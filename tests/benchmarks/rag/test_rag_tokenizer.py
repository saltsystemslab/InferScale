from __future__ import annotations

import pytest

from benchmarks.rag.tokenizer import load_rag_tokenizer
from inferscale.v1.kv import encoder


@pytest.fixture
def engine_tokenizer_loader(monkeypatch):
    def install(outcome):
        def load(model: str):
            if isinstance(outcome, Exception):
                raise outcome
            return f"{outcome} for {model}"

        monkeypatch.setattr(encoder, "load_encoder_tokenizer", load)

    return install


@pytest.fixture
def transformers_tokenizers(monkeypatch) -> list[str]:
    transformers = pytest.importorskip("transformers", reason="the fallback loads AutoTokenizer")
    loaded: list[str] = []

    def from_pretrained(model: str) -> str:
        loaded.append(model)
        return f"transformers tokenizer for {model}"

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    return loaded


def test_estimates_fall_back_to_the_transformers_tokenizer_with_a_warning(
    engine_tokenizer_loader, transformers_tokenizers, rag_log
):
    engine_tokenizer_loader(ModuleNotFoundError("No module named 'vllm'"))

    tokenizer = load_rag_tokenizer("some/model", allow_transformers_fallback=True)

    assert tokenizer == "transformers tokenizer for some/model"
    assert transformers_tokenizers == ["some/model"]
    assert rag_log == [
        (
            "WARNING",
            "vLLM is unavailable (No module named 'vllm'); falling back to transformers "
            "AutoTokenizer for model some/model. Estimates only: token counts may differ "
            "slightly from the engine.",
        )
    ]
