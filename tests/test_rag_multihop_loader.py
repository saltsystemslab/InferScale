from __future__ import annotations

import pytest

from rag_bench.data_types import doc_id_for_url
from rag_bench.datasets import get_dataset
from rag_bench.datasets.multihop_rag import load_multihop_rag
from rag_test_utils import (
    FACT_B,
    multihop_corpus_records,
    multihop_query_records,
    write_multihop_files,
)


def test_registry_exposes_multihoprag_spec() -> None:
    spec = get_dataset("multihoprag")

    assert spec.name == "multihoprag"
    assert spec.chunking == "token-window"
    assert set(spec.download_urls) == {"corpus.json", "MultiHopRAG.json"}
    with pytest.raises(ValueError, match="Unknown RAG dataset"):
        get_dataset("nope")


def test_loader_parses_fixture_and_resolves_evidence(tmp_path) -> None:
    data_dir = write_multihop_files(tmp_path / "data")

    docs, queries = load_multihop_rag(data_dir)

    assert [doc.title for doc in docs] == [
        "Alpha Corp stuns the energy market",
        "Beta Industries posts record quarter",
        "Gamma LLC slips its launch window",
    ]
    assert all(doc.doc_id == doc_id_for_url(doc.url) for doc in docs)
    assert docs[1].author is None

    assert [query.query_id for query in queries] == ["q0000", "q0001", "q0002", "q0003"]
    assert [query.question_type for query in queries] == [
        "inference_query",
        "comparison_query",
        "temporal_query",
        "null_query",
    ]
    inference = queries[0]
    assert {ref.doc_id for ref in inference.evidence} == {docs[0].doc_id, docs[1].doc_id}
    assert queries[3].is_null and queries[3].evidence == ()


def test_evidence_resolves_by_title_when_url_is_missing(tmp_path) -> None:
    data_dir = write_multihop_files(tmp_path / "data")

    docs, queries = load_multihop_rag(data_dir)
    temporal = queries[2]

    assert len(temporal.evidence) == 1
    assert temporal.evidence[0].doc_id == docs[1].doc_id
    assert temporal.evidence[0].fact == FACT_B


def test_unresolvable_evidence_raises(tmp_path) -> None:
    queries = multihop_query_records()
    queries[0]["evidence_list"][0]["url"] = "https://example.com/not-in-corpus"
    queries[0]["evidence_list"][0]["title"] = "Not a corpus title"
    data_dir = write_multihop_files(tmp_path / "data", queries=queries)

    with pytest.raises(ValueError, match="does not resolve to any corpus document"):
        load_multihop_rag(data_dir)


def test_null_query_with_evidence_raises(tmp_path) -> None:
    queries = multihop_query_records()
    queries[3]["evidence_list"] = list(queries[0]["evidence_list"])
    data_dir = write_multihop_files(tmp_path / "data", queries=queries)

    with pytest.raises(ValueError, match="null_query but has evidence"):
        load_multihop_rag(data_dir)


def test_answerable_query_without_evidence_raises(tmp_path) -> None:
    queries = multihop_query_records()
    queries[1]["evidence_list"] = []
    data_dir = write_multihop_files(tmp_path / "data", queries=queries)

    with pytest.raises(ValueError, match="has no evidence"):
        load_multihop_rag(data_dir)


def test_unknown_question_type_raises(tmp_path) -> None:
    queries = multihop_query_records()
    queries[0]["question_type"] = "mystery_query"
    data_dir = write_multihop_files(tmp_path / "data", queries=queries)

    with pytest.raises(ValueError, match="unknown question_type"):
        load_multihop_rag(data_dir)


def test_duplicate_corpus_urls_raise(tmp_path) -> None:
    corpus = multihop_corpus_records()
    corpus[2]["url"] = corpus[0]["url"]
    data_dir = write_multihop_files(tmp_path / "data", corpus=corpus)

    with pytest.raises(ValueError, match="share the url"):
        load_multihop_rag(data_dir)


def test_missing_corpus_field_raises(tmp_path) -> None:
    corpus = multihop_corpus_records()
    del corpus[1]["published_at"]
    data_dir = write_multihop_files(tmp_path / "data", corpus=corpus)

    with pytest.raises(ValueError, match="missing required field 'published_at'"):
        load_multihop_rag(data_dir)


def test_missing_files_raise_with_download_hint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="setup_data.sh"):
        load_multihop_rag(tmp_path / "does-not-exist")
