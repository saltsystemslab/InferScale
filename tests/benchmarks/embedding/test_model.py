from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.embedding import model as embedding_model
from benchmarks.embedding.model import QwenEmbedder

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

SETTINGS = dict(
    model="Qwen/Qwen3-Embedding-0.6B",
    revision="main",
    device="cpu",
    dtype="auto",
    attention="sdpa",
    max_length=32768,
)


class LoadedModel:
    def __init__(self, journal: list[Any]) -> None:
        self.journal = journal
        self.config = SimpleNamespace(hidden_size=1024, _commit_hash="resolved-commit")

    def to(self, device: Any) -> LoadedModel:
        self.journal.append(("model moved to", str(device)))
        return self

    def eval(self) -> LoadedModel:
        self.journal.append("model set to eval")
        return self


@pytest.fixture
def journal(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    journal: list[Any] = []
    seconds = itertools.count(start=100.0, step=2.5)

    def read_clock() -> float:
        journal.append("clock read")
        return next(seconds)

    def load_tokenizer(name: str, **options: Any) -> str:
        journal.append(("tokenizer loaded", name, options))
        return "tokenizer"

    def load_model(name: str, **options: Any) -> LoadedModel:
        journal.append(("model loaded", name, options))
        return LoadedModel(journal)

    monkeypatch.setattr(embedding_model.time, "perf_counter", read_clock)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", load_model)
    return journal


@pytest.fixture
def cuda(monkeypatch: pytest.MonkeyPatch, journal: list[Any]) -> None:
    def record(name: str) -> Any:
        def call(*arguments: Any) -> None:
            journal.append((name, *map(str, arguments)))

        return call

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", record("device selected"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", record("synchronized"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")


def test_gpu_load_time_ends_after_the_device_has_finished_its_work(
    journal: list[Any], cuda: None
) -> None:
    embedder = QwenEmbedder(
        **{**SETTINGS, "device": "cuda:1", "attention": "eager", "max_length": 512}
    )

    assert journal == [
        ("device selected", "cuda:1"),
        "clock read",
        ("tokenizer loaded", "Qwen/Qwen3-Embedding-0.6B", {"revision": "main"}),
        (
            "model loaded",
            "Qwen/Qwen3-Embedding-0.6B",
            {"revision": "main", "torch_dtype": torch.float16, "attn_implementation": "eager"},
        ),
        ("model moved to", "cuda:1"),
        "model set to eval",
        ("synchronized", "cuda:1"),
        "clock read",
    ]
    assert embedder.metadata["device"] == "cuda:1"
    assert embedder.metadata["dtype"] == "float16"
    assert (embedder.max_length, embedder.metadata["max_length"]) == (512, 512)
    assert embedder.metadata["load_seconds"] == 2.5
    assert embedder.metadata["gpu"] == "Test GPU"


class WordTokenizer:
    def __init__(self) -> None:
        self.batches: list[tuple[int, dict[str, Any]]] = []

    def __call__(self, texts: list[str], **options: Any) -> dict[str, Any]:
        self.batches.append((len(texts), options))
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


def embedder_with(tokenizer: Any, *, max_length: int) -> QwenEmbedder:
    embedder = QwenEmbedder.__new__(QwenEmbedder)
    embedder.tokenizer = tokenizer
    embedder.max_length = max_length
    return embedder


def test_token_lengths_are_counted_without_truncation_in_batches_of_256() -> None:
    tokenizer = WordTokenizer()
    texts = [" ".join(["word"] * (1 + index % 4)) for index in range(600)]

    lengths = embedder_with(tokenizer, max_length=4).token_lengths(texts)

    assert lengths == [1 + index % 4 for index in range(600)]
    assert tokenizer.batches == [
        (256, {"truncation": False}),
        (256, {"truncation": False}),
        (88, {"truncation": False}),
    ]
    assert embedding_model.TOKENIZE_BATCH_SIZE == 256


def embedder_of_width(hidden_size: int) -> QwenEmbedder:
    embedder = QwenEmbedder.__new__(QwenEmbedder)
    embedder.torch = torch
    embedder.model = SimpleNamespace(config=SimpleNamespace(hidden_size=hidden_size))
    return embedder


def test_validation_reports_the_shape_before_the_values() -> None:
    not_finite = torch.nn.functional.normalize(torch.ones(2, 32), dim=1)
    not_finite[0, 0] = float("nan")
    embedder = embedder_of_width(32)

    with pytest.raises(ValueError) as wrong_count:
        embedder.validate(not_finite, 3)
    assert str(wrong_count.value) == "Invalid embedding shape: torch.Size([2, 32])"
    with pytest.raises(ValueError, match=r"Invalid embedding shape: torch.Size\(\[2, 32\]\)"):
        embedder_of_width(16).validate(not_finite, 2)
    with pytest.raises(ValueError) as non_finite:
        embedder.validate(not_finite, 2)
    assert str(non_finite.value) == "Non-finite embeddings"
    infinite = torch.full((2, 32), float("inf"))
    with pytest.raises(ValueError, match="Non-finite embeddings"):
        embedder.validate(infinite, 2)
