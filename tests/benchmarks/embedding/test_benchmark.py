import json

import pytest

from benchmarks.embedding.corpus import load_corpus
from benchmarks.embedding.run import measure


def catalog(path, sample="sample-1", texts=(" A fact. ", "Repeated", "Repeated"), **metadata):
    payload = {"version": 5, "sample_id": sample, "model": "extractor",
               "embedding_model": "text-embedding-3-small", **metadata,
               "facts": [{"id": str(i), "sample_id": sample, "text": text}
                         for i, text in enumerate(texts)]}
    path.write_text(json.dumps(payload))


def test_corpus_preserves_exact_text_and_duplicates_with_stable_selection(tmp_path):
    catalog(tmp_path / "sample.json")
    texts, info = load_corpus(tmp_path)
    assert sorted(texts) == [" A fact. ", "Repeated", "Repeated"]
    assert info["unique_texts"] == 2
    assert info["selected_facts"] == 3
    assert load_corpus(tmp_path)[1] == info
    subset, manifest = load_corpus(tmp_path, max_facts=2)
    assert subset == texts[:2]
    assert manifest["available_facts"] == 3
    assert manifest["ordered_texts_sha256"] != info["ordered_texts_sha256"]


@pytest.mark.parametrize("change, message", [
    ({"sample": "sample-1"}, "duplicate sample_id"),
    ({"sample": "sample-2", "model": "different"}, "Mixed"),
    ({"sample": "sample-2", "version": 4}, "version 5"),
    ({"sample": "sample-2", "texts": [""]}, "Invalid fact text"),
])
def test_rejects_invalid_or_ambiguous_catalogs(tmp_path, change, message):
    catalog(tmp_path / "a.json")
    catalog(tmp_path / "b.json", **change)
    with pytest.raises(ValueError, match=message):
        load_corpus(tmp_path)


def test_empty_corpus_and_missing_leaf_fail(tmp_path):
    with pytest.raises(ValueError, match="No catalogs"):
        load_corpus(tmp_path)
    catalog(tmp_path / "empty.json", texts=[])
    with pytest.raises(ValueError, match="no facts"):
        load_corpus(tmp_path)


def test_measure_excludes_warmup_and_counts_partial_batches():
    events = []

    class FakeEmbedder:
        def encode(self, texts):
            events.append(("encode", list(texts)))
            return texts

        def synchronize(self):
            events.append("sync")

        def validate(self, vectors, count):
            events.append("validate")
            assert len(vectors) == count

        def reset_peak_memory(self):
            events.append("reset")

        def peak_memory_bytes(self):
            return 123

    ticks = iter([0, 0.1, 1, 1.2, 2, 2.1, 3, 3.2])

    def clock():
        events.append("clock")
        return next(ticks)

    result = measure(FakeEmbedder(), ["a", "b", "c"], [2, 3, 4],
                     batch_size=2, warmup=1, repeats=2, clock=clock)
    assert result["total_facts"] == 6
    assert result["total_tokens"] == 18
    assert result["facts_per_second"] == pytest.approx(10)
    assert result["tokens_per_second"] == pytest.approx(30)
    assert result["amortized_ms_per_fact"] == pytest.approx(100)
    assert result["batch_latency_ms"]["p50"] == pytest.approx(150)
    assert [row["facts"] for row in result["batches"]] == [2, 1, 2, 1]
    assert events[:4] == [("encode", ["a", "b"]), "sync", "validate", "reset"]
    assert events[4:10] == ["sync", "clock", ("encode", ["a", "b"]), "sync", "clock", "validate"]


def test_qwen_pooling_is_batch_invariant_without_downloading_weights():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3Config, Qwen3Model
    from benchmarks.embedding.model import QwenEmbedder

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            from transformers import BatchEncoding
            rows = [[int(token) for token in text.split()] for text in texts]
            width = max(map(len, rows))
            return BatchEncoding({
                "input_ids": torch.tensor([[0] * (width - len(row)) + row for row in rows]),
                "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row)
                                                for row in rows]),
            })

    embedder = QwenEmbedder.__new__(QwenEmbedder)
    embedder.torch = torch
    embedder.device = torch.device("cpu")
    embedder.tokenizer = Tokenizer()
    embedder.model = Qwen3Model(Qwen3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16,
    )).eval()
    batched = embedder.encode(["1 2", "3 4 5 6"])
    individual = torch.cat([embedder.encode([text]) for text in ["1 2", "3 4 5 6"]])
    embedder.validate(batched, 2)
    assert torch.allclose(batched, individual, atol=1e-5)


def test_qwen_default_cuda_resolves_current_device_before_selecting_it(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from benchmarks.embedding.model import QwenEmbedder

    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "set_device",
        lambda device: selected.append(torch.cuda._utils._get_device_index(device)),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    model = Mock(config=SimpleNamespace(hidden_size=1024, _commit_hash="test-revision"))
    model.to.return_value = model
    model.eval.return_value = model
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **kw: model)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: Mock())

    embedder = QwenEmbedder(
        model="Qwen/Qwen3-Embedding-0.6B", revision="main", device="cuda",
        dtype="auto", attention="sdpa", max_length=32768,
    )

    assert selected == [2]
    assert embedder.metadata["device"] == "cuda:2"
    assert embedder.metadata["dtype"] == "bfloat16"
    model.to.assert_called_once_with(torch.device("cuda:2"))
