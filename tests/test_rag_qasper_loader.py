from __future__ import annotations

import pytest

from rag_bench.data_types import doc_id_for_url
from rag_bench.datasets import get_dataset
from rag_bench.datasets.qasper import UNANSWERABLE_TEXT, load_qasper
from rag_test_utils import (
    QASPER_EVIDENCE_RESULTS,
    QASPER_EVIDENCE_SURVEY,
    qasper_test_records,
    write_qasper_files,
)


def test_registry_exposes_qasper_spec() -> None:
    spec = get_dataset("qasper")

    assert spec.name == "qasper"
    assert spec.chunking == "token-window"
    assert UNANSWERABLE_TEXT in spec.prompt_profile.answer_instruction
    assert "research papers" in spec.prompt_profile.system_prompt


def test_loader_parses_papers_and_questions(tmp_path) -> None:
    data_dir = write_qasper_files(tmp_path / "data")

    docs, queries = load_qasper(data_dir)

    assert [doc.title for doc in docs] == [
        "Paper One: Attention Study",
        "Paper Two: Datasets Survey",
    ]
    first = docs[0]
    assert first.url == "https://arxiv.org/abs/1601.00001"
    assert first.doc_id == doc_id_for_url(first.url)
    assert first.source == "arXiv"
    assert first.published_at == "2016-01"
    assert docs[1].published_at == "2017-07"
    assert first.body.startswith("We study attention mechanisms")
    assert "Introduction" in first.body
    assert QASPER_EVIDENCE_RESULTS in first.body

    assert [query.query_id for query in queries] == [
        "q-extractive",
        "q-boolean",
        "q-unanswerable",
    ]


def test_extractive_references_join_spans_and_keep_all_annotations(tmp_path) -> None:
    data_dir = write_qasper_files(tmp_path / "data")

    _, queries = load_qasper(data_dir)
    extractive = queries[0]

    assert extractive.gold_answers == ("42.0 F1, on the probing benchmark", "42 F1")
    assert extractive.gold_answer == "42.0 F1, on the probing benchmark"
    assert extractive.question_type == "extractive"


def test_boolean_and_unanswerable_references(tmp_path) -> None:
    data_dir = write_qasper_files(tmp_path / "data")

    docs, queries = load_qasper(data_dir)
    boolean = queries[1]
    unanswerable = queries[2]

    assert boolean.gold_answers == ("Yes", "No")
    assert boolean.question_type == "boolean"
    assert unanswerable.gold_answers == (UNANSWERABLE_TEXT, "Twelve annotators")
    assert unanswerable.question_type == "unanswerable"
    # The answering annotation still contributes evidence from its paper.
    assert [ref.fact for ref in unanswerable.evidence] == [QASPER_EVIDENCE_SURVEY]
    assert unanswerable.evidence[0].doc_id == docs[1].doc_id


def test_evidence_is_deduped_and_float_selected_filtered(tmp_path) -> None:
    data_dir = write_qasper_files(tmp_path / "data")

    docs, queries = load_qasper(data_dir)
    extractive = queries[0]

    assert [ref.fact for ref in extractive.evidence] == [QASPER_EVIDENCE_RESULTS]
    assert extractive.evidence[0].doc_id == docs[0].doc_id


def test_annotation_without_answer_raises(tmp_path) -> None:
    records = qasper_test_records()
    records["1601.00001"]["qas"][0]["answers"][0]["answer"] = {
        "unanswerable": False,
        "extractive_spans": [],
        "yes_no": None,
        "free_form_answer": "",
        "evidence": [],
        "highlighted_evidence": [],
    }
    data_dir = write_qasper_files(tmp_path / "data", records)

    with pytest.raises(ValueError, match="without an answer"):
        load_qasper(data_dir)


def test_duplicate_question_ids_raise(tmp_path) -> None:
    records = qasper_test_records()
    records["1707.99999"]["qas"][0]["question_id"] = "q-extractive"
    data_dir = write_qasper_files(tmp_path / "data", records)

    with pytest.raises(ValueError, match="duplicate question_id"):
        load_qasper(data_dir)


def test_missing_file_raises_with_setup_hint(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="RAG_DATASET=qasper"):
        load_qasper(tmp_path / "does-not-exist")


def test_unversioned_paper_id_published_is_unknown(tmp_path) -> None:
    records = qasper_test_records()
    records["not-an-arxiv-id"] = records.pop("1707.99999")
    data_dir = write_qasper_files(tmp_path / "data", records)

    docs, _ = load_qasper(data_dir)

    by_title = {doc.title: doc for doc in docs}
    assert by_title["Paper Two: Datasets Survey"].published_at == "unknown"
