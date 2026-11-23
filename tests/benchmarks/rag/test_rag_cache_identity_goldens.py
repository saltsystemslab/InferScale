from __future__ import annotations

import pytest
from rag_fixtures import write_multihop_files

from benchmarks.rag.chunking import corpus_fingerprint, render_document
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import ABSTENTION_QUESTION_TYPES, RagChunk, doc_id_for_url
from benchmarks.rag.datasets.multihop_rag import load_multihop_rag
from benchmarks.rag.kv_cache import RagChunkCache

FIXTURE_CORPUS_FINGERPRINT = "c948f981ee8b1503483e"


@pytest.fixture
def docs(tmp_path):
    documents, _ = load_multihop_rag(write_multihop_files(tmp_path / "data"))
    return documents


def test_document_id_is_a_stable_short_hash_of_the_stripped_url():
    assert doc_id_for_url("https://example.com/alpha-corp-solar") == "d55c91bbf3bb8"
    assert doc_id_for_url("  https://example.com/alpha-corp-solar\n") == "d55c91bbf3bb8"
    assert doc_id_for_url("https://example.com/beta-industries-quarter") == "d74c82fbcf8ff"
    with pytest.raises(ValueError, match="Document URL must be non-empty to derive a doc id"):
        doc_id_for_url("   ")


def test_rendered_document_puts_its_metadata_before_the_stripped_body(docs):
    assert render_document(docs[1]) == (
        "Title: Beta Industries posts record quarter\n"
        "Source: Example Business Daily\n"
        "Published: 2023-10-15T12:30:00+00:00\n"
        "Category: business\n\n"
        f"{docs[1].body}\n"
    )


def test_corpus_fingerprint_is_stable_for_the_same_documents_in_the_same_order(docs):
    assert corpus_fingerprint(docs) == FIXTURE_CORPUS_FINGERPRINT
    assert corpus_fingerprint([]) == "e3b0c44298fc1c149afb"
    assert corpus_fingerprint(docs[:2]) != corpus_fingerprint(docs[1::-1])


def test_kv_cache_layout_is_keyed_by_model_encoding_settings_and_corpus(docs, tmp_path):
    config = RagBenchConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        chunk_size=512,
        context_window=2,
        kv_max_position=4096,
        kv_chunk_cache_root=tmp_path / "kv",
    )

    cache = RagChunkCache.for_config(config, docs)

    assert cache.directory == (
        tmp_path
        / "kv"
        / "v1"
        / "meta-llama__Llama-3.1-8B-Instruct"
        / "bfloat16-c512-w2-p4096"
        / FIXTURE_CORPUS_FINGERPRINT
    )
    assert cache.chunk_path("d55c91bbf3bb8:3") == cache.directory / "chunks" / "d55c91bbf3bb8_3.pt"
    assert cache.tables_path(16) == cache.directory / "tables-scaffold-b16.pt"
    chunk = RagChunk("d55c91bbf3bb8:3", "d55c91bbf3bb8", 3, [1, 2, 3], "abc")
    identity = {
        "dataset": "multihoprag",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "dtype": "bfloat16",
        "chunk_size": 512,
        "context_window": 2,
        "max_position": 4096,
        "corpus_fingerprint": FIXTURE_CORPUS_FINGERPRINT,
    }
    assert cache.key.meta() == identity
    assert cache.key.chunk_meta(chunk) == {
        **identity,
        "chunk_id": "d55c91bbf3bb8:3",
        "token_count": 3,
    }
    assert cache.key.tables_meta(block_size=16) == {**identity, "block_size": 16}


def test_abstention_is_the_gold_behavior_of_null_and_unanswerable_questions():
    assert sorted(ABSTENTION_QUESTION_TYPES) == ["null_query", "unanswerable"]
