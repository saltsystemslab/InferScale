from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from benchmarks.common.vector_types import SearchHit
from benchmarks.rag import metrics
from benchmarks.rag.data_types import EvidenceRef, RagQuery
from benchmarks.rag.datasets.multihop_rag import INSUFFICIENT_ANSWER_TEXT
from benchmarks.rag.datasets.qasper import UNANSWERABLE_TEXT
from benchmarks.rag.metrics import predicted_insufficient, retrieval_metrics_for_query


def hit(doc_id, chunk_index, rank: int) -> SearchHit:
    return SearchHit(
        id=f"{doc_id}:{chunk_index}",
        payload={"doc_id": doc_id, "chunk_index": chunk_index},
        score=0.0,
        distance=0.0,
        rank=rank,
    )


def query_with_evidence(*facts_by_doc: tuple[str, str]) -> RagQuery:
    return RagQuery(
        query_id="q0",
        question="What happened?",
        gold_answers=("Something",),
        question_type="inference_query",
        evidence=tuple(
            EvidenceRef(doc_id=doc_id, title=doc_id, url=doc_id, fact=fact)
            for doc_id, fact in facts_by_doc
        ),
    )


def test_every_punctuation_mapping_can_still_match_after_compatibility_normalization():
    rewritten_before_the_table_is_applied = [
        f"U+{code_point:04X}"
        for code_point in metrics._FACT_PUNCTUATION_TO_ASCII
        if unicodedata.normalize("NFKC", chr(code_point)) != chr(code_point)
    ]

    assert rewritten_before_the_table_is_applied == []


def test_metrics_source_spells_its_unicode_punctuation_as_escapes():
    assert Path(metrics.__file__).read_bytes().isascii()


@pytest.mark.parametrize("abstention_answer", [INSUFFICIENT_ANSWER_TEXT, UNANSWERABLE_TEXT])
def test_the_answer_each_dataset_prompt_asks_for_when_unsure_counts_as_abstention(
    abstention_answer,
):
    assert predicted_insufficient(abstention_answer) is True
    assert predicted_insufficient(f"{abstention_answer}.") is True
    assert predicted_insufficient(f"{abstention_answer.upper()} - the excerpts do not say") is True


def test_retrieval_metrics_count_each_retrieved_document_once():
    query = query_with_evidence(("dA", "alpha fact"), ("dB", "beta fact"), ("dA", "more alpha"))
    hits = [hit("dX", 0, 1), hit("dA", 1, 2), hit("dA", 0, 3), hit("dX", 1, 4)]
    texts = {"dA:0": "an alpha f", "dA:1": "act and more alpha", "dX:0": "beta fact elsewhere"}

    assert retrieval_metrics_for_query(query, hits, texts) == {
        "evidence_doc_count": 2,
        "retrieved_chunk_count": 4,
        "retrieved_doc_count": 2,
        "evidence_recall_at_k": 0.5,
        "evidence_full_recall_at_k": 0.0,
        "evidence_hit_any_at_k": 1.0,
        "doc_mrr_at_k": 0.5,
        "fact_recall_at_k": 2 / 3,
    }
