from __future__ import annotations

import pytest

from locomo_jasper_bench.vector_types import SearchHit
from rag_bench.data_types import EvidenceRef, RagQuery
from rag_bench.metrics import (
    answer_metrics,
    exact_match,
    normalize_answer,
    normalize_passage_text,
    predicted_insufficient,
    retrieval_metrics_for_query,
    substring_match,
    token_f1,
)


def _hit(chunk_id: str, doc_id: str, chunk_index: int, rank: int) -> SearchHit:
    return SearchHit(
        id=chunk_id,
        payload={"chunk_id": chunk_id, "doc_id": doc_id, "chunk_index": chunk_index},
        score=float(-rank),
        distance=float(rank),
        rank=rank,
    )


def _query(evidence: tuple[EvidenceRef, ...], question_type: str = "inference_query") -> RagQuery:
    return RagQuery(
        query_id="q0",
        question="What happened?",
        gold_answers=("Something",),
        question_type=question_type,
        evidence=evidence,
    )


def test_normalize_answer_strips_articles_punctuation_and_case() -> None:
    assert normalize_answer("The Apple, Inc.") == "apple inc"
    assert normalize_answer("  a  QUICK   fox! ") == "quick fox"


def test_token_f1_hand_computed() -> None:
    result = token_f1("apple inc", "apple")

    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(1.0)
    assert result["f1"] == pytest.approx(2 / 3)
    assert token_f1("banana", "apple")["f1"] == 0.0
    assert token_f1("", "")["f1"] == 1.0
    assert token_f1("", "apple")["f1"] == 0.0


def test_exact_and_substring_match() -> None:
    assert exact_match("The answer is: Apple!", "apple") is False
    assert exact_match("Apple.", "the apple") is True
    assert substring_match("It was Apple, Inc. that won", "apple inc") is True
    assert substring_match("It was Banana", "apple") is False


def test_predicted_insufficient_variants() -> None:
    assert predicted_insufficient("Insufficient information.") is True
    assert predicted_insufficient("insufficient information") is True
    assert predicted_insufficient("INSUFFICIENT INFORMATION!") is True
    assert predicted_insufficient("Unanswerable") is True
    assert predicted_insufficient("unanswerable from the excerpts") is True
    assert predicted_insufficient("The information is insufficient") is False
    assert predicted_insufficient("Apple") is False


def test_answer_metrics_bundle_accepts_a_bare_string() -> None:
    metrics = answer_metrics("Insufficient information.", "Insufficient information.")

    assert metrics["exact_match"] is True
    assert metrics["f1"] == 1.0
    assert metrics["substring_match"] is True
    assert metrics["predicted_insufficient"] is True


def test_answer_metrics_take_the_best_over_references() -> None:
    metrics = answer_metrics("42 F1", ["42.0 F1, on the probing benchmark", "42 F1"])

    assert metrics["exact_match"] is True
    assert metrics["f1"] == 1.0
    assert metrics["substring_match"] is True

    partial = answer_metrics("apple inc", ["banana", "apple"])
    assert partial["exact_match"] is False
    assert partial["f1"] == pytest.approx(2 / 3)
    assert partial["recall"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="at least one gold answer"):
        answer_metrics("x", [])


def test_normalize_passage_text_maps_unicode_punctuation() -> None:
    fancy = "It\u2019s a \u201cbig\u201d win \u2014 maybe\u2026"
    plain = "it's a \"big\" win - maybe..."

    assert normalize_passage_text(fancy) == plain
    assert normalize_passage_text("A  \n\t B") == "a b"


def test_retrieval_metrics_partial_and_full_recall() -> None:
    evidence = (
        EvidenceRef(doc_id="dA", title="A", url="uA", fact="alpha fact"),
        EvidenceRef(doc_id="dB", title="B", url="uB", fact="beta fact"),
    )
    texts = {"dA:0": "before alpha fact after", "dB:0": "around beta fact here", "dX:0": "noise"}

    partial = retrieval_metrics_for_query(
        _query(evidence),
        [_hit("dX:0", "dX", 0, 1), _hit("dA:0", "dA", 0, 2)],
        texts,
    )
    assert partial["evidence_recall_at_k"] == pytest.approx(0.5)
    assert partial["evidence_full_recall_at_k"] == 0.0
    assert partial["evidence_hit_any_at_k"] == 1.0
    assert partial["doc_mrr_at_k"] == pytest.approx(0.5)
    assert partial["fact_recall_at_k"] == pytest.approx(0.5)
    assert partial["retrieved_doc_count"] == 2

    full = retrieval_metrics_for_query(
        _query(evidence),
        [_hit("dA:0", "dA", 0, 1), _hit("dB:0", "dB", 0, 2)],
        texts,
    )
    assert full["evidence_recall_at_k"] == 1.0
    assert full["evidence_full_recall_at_k"] == 1.0
    assert full["doc_mrr_at_k"] == 1.0
    assert full["fact_recall_at_k"] == 1.0


def test_fact_spanning_adjacent_chunks_counts_when_both_retrieved() -> None:
    evidence = (EvidenceRef(doc_id="dA", title="A", url="uA", fact="split across chunks"),)
    texts = {"dA:0": "the fact is split acr", "dA:1": "oss chunks in the body"}

    both = retrieval_metrics_for_query(
        _query(evidence),
        [_hit("dA:1", "dA", 1, 1), _hit("dA:0", "dA", 0, 2)],
        texts,
    )
    assert both["fact_recall_at_k"] == 1.0

    only_one = retrieval_metrics_for_query(
        _query(evidence),
        [_hit("dA:0", "dA", 0, 1)],
        texts,
    )
    assert only_one["fact_recall_at_k"] == 0.0
    assert only_one["evidence_recall_at_k"] == 1.0


def test_unicode_fact_matches_ascii_chunk_text() -> None:
    evidence = (
        EvidenceRef(doc_id="dA", title="A", url="uA", fact="it\u2019s a \u201cwin\u201d \u2013 truly"),
    )
    texts = {"dA:0": "they said it's a \"win\" - truly impressive"}

    metrics = retrieval_metrics_for_query(_query(evidence), [_hit("dA:0", "dA", 0, 1)], texts)

    assert metrics["fact_recall_at_k"] == 1.0


def test_null_query_returns_none() -> None:
    assert retrieval_metrics_for_query(_query((), "null_query"), [], {}) is None


def test_missing_doc_id_payload_raises() -> None:
    evidence = (EvidenceRef(doc_id="dA", title="A", url="uA", fact="alpha"),)
    bad_hit = SearchHit(id="x", payload={}, score=0.0, distance=0.0, rank=1)

    with pytest.raises(RuntimeError, match="no doc_id payload"):
        retrieval_metrics_for_query(_query(evidence), [bad_hit], {})
