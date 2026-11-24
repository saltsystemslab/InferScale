from __future__ import annotations

from typing import Any

import pytest

from inferscale.v1.embedding.local import LocalEmbedder

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

VOCABULARY_SIZE = 32
HIDDEN_SIZE = 32
TEXTS = ["Alice adopted Rex", "Bob moved to Lisbon in March", "tea"]


class WordTokenizer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, texts: list[str], **options: Any) -> Any:
        self.batch_sizes.append(len(texts))
        rows = [[sum(word.encode()) % VOCABULARY_SIZE for word in text.split()] for text in texts]
        width = max(map(len, rows))
        return transformers.BatchEncoding(
            {
                "input_ids": torch.tensor([[0] * (width - len(row)) + row for row in rows]),
                "attention_mask": torch.tensor(
                    [[0] * (width - len(row)) + [1] * len(row) for row in rows]
                ),
            }
        )


@pytest.fixture
def tokenizer(monkeypatch: pytest.MonkeyPatch) -> WordTokenizer:
    tokenizer = WordTokenizer()
    tiny_model = transformers.Qwen3Model(
        transformers.Qwen3Config(
            vocab_size=VOCABULARY_SIZE,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=16,
        )
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda name, **options: tokenizer
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", lambda name, **options: tiny_model
    )
    return tokenizer


def test_documents_and_queries_get_the_same_unit_vectors_whatever_the_batch(
    tokenizer: WordTokenizer,
) -> None:
    embedder = LocalEmbedder(model="tiny-qwen", device="cpu", batch_size=2)

    batched = torch.tensor(embedder.embed_documents(TEXTS))
    one_at_a_time = torch.tensor([embedder.embed_query(text) for text in TEXTS])

    assert embedder.dimensions == HIDDEN_SIZE
    assert batched.shape == (len(TEXTS), HIDDEN_SIZE)
    assert torch.allclose(batched, one_at_a_time, atol=1e-5)
    assert torch.allclose(batched.norm(dim=1), torch.ones(len(TEXTS)), atol=1e-4)
    warm_up, *measured = tokenizer.batch_sizes
    assert (warm_up, measured) == (1, [2, 1, 1, 1, 1])
