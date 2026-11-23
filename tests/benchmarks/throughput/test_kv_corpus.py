from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.throughput import kv_corpus
from benchmarks.memory.throughput.kv_corpus import PrecomputedCorpus, precompute_corpus
from benchmarks.memory.throughput.projection import CorpusFootprint
from inferscale.v1.kv.encoder import TOKENIZER_PARITY_PROBE_TEXT

PROBE_TOKEN_IDS = [4, 5, 6]


class FakeComposer:
    def __init__(self, sample_id: str, fact_token_counts: list[int]) -> None:
        self.chunks = {
            f"{sample_id}-fact-{index}": SimpleNamespace(token_ids=[0] * token_count)
            for index, token_count in enumerate(fact_token_counts)
        }
        self.header_chunk: Any = SimpleNamespace(token_ids=[1, 2, 3])
        self.footer_chunk: Any = SimpleNamespace(token_ids=[9])
        self.stored_chunk_bytes = 1000 * len(fact_token_counts)
        self.encoder = SimpleNamespace(tokenizer=f"tokenizer-of-{sample_id}")


class FakePrecompute:
    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.composers: list[FakeComposer] = []

    def precompute(self, sample: Any, hits: list[str]) -> tuple[FakeComposer, bool]:
        self.events.append(("precompute", sample.sample_id, hits))
        composer = FakeComposer(sample.sample_id, [2] * len(hits))
        self.composers.append(composer)
        return composer, False

    def release_model(self) -> None:
        self.events.append("release_model")
        for composer in self.composers:
            composer.encoder.tokenizer = None


@pytest.fixture
def corpus_events(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    events: list[Any] = []
    monkeypatch.setattr(
        kv_corpus, "fact_catalog_hits", lambda facts: [f"hit of {fact}" for fact in facts]
    )
    monkeypatch.setattr(
        kv_corpus,
        "encode_text_no_special",
        lambda tokenizer, text: (events.append(("probe", tokenizer, text)), PROBE_TOKEN_IDS)[1],
    )
    monkeypatch.setattr(
        kv_corpus, "release_cuda_memory", lambda: events.append("release_cuda_memory")
    )
    return events


def precompute_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: list[Any], sample_ids: list[str]
) -> PrecomputedCorpus:
    precompute = FakePrecompute(events)
    monkeypatch.setattr(kv_corpus, "SamplePrecompute", lambda config: precompute)
    samples = [SimpleNamespace(sample_id=sample_id) for sample_id in sample_ids]
    facts_by_sample = {
        sample_id: tuple(f"{sample_id}-fact-{index}" for index in range(position + 1))
        for position, sample_id in enumerate(sample_ids)
    }
    return precompute_corpus(make_throughput_config(tmp_path), samples, facts_by_sample)


def test_corpus_keeps_the_tokenizer_it_read_before_the_encoder_weights_were_released(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corpus_events: list[Any], ticking_clock
) -> None:
    ticking_clock(corpus_events)

    corpus = precompute_samples(monkeypatch, tmp_path, corpus_events, ["a", "b"])

    assert corpus_events == [
        "clock",
        ("precompute", "a", ["hit of a-fact-0"]),
        ("precompute", "b", ["hit of b-fact-0", "hit of b-fact-1"]),
        ("probe", "tokenizer-of-a", TOKENIZER_PARITY_PROBE_TEXT),
        "release_model",
        "release_cuda_memory",
        "clock",
    ]
    assert corpus.tokenizer == "tokenizer-of-a"
    assert corpus.composer("a").encoder.tokenizer is None
    assert corpus.probe_token_ids == PROBE_TOKEN_IDS
    assert corpus.precompute_time_s == 3.0
    assert list(corpus.composers) == ["a", "b"]
    assert corpus.footprint == CorpusFootprint(
        fact_count=1, token_count=2, kv_bytes=1000, scaffold_token_count=4
    )
