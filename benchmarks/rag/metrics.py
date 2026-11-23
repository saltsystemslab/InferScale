from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from benchmarks.common.vector_types import SearchHit
from benchmarks.rag.data_types import RagQuery

_SQUAD_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_SQUAD_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

NORMALIZED_ABSTENTION_PREFIXES = ("insufficient information", "unanswerable")

_HYPHEN_AND_DASH_VARIANTS = ("\u2010", "\u2012", "\u2013", "\u2014", "\u2015")
_CURLY_SINGLE_QUOTES = ("\u2018", "\u2019", "\u201a")
_CURLY_DOUBLE_QUOTES = ("\u201c", "\u201d", "\u201e")
_FACT_PUNCTUATION_TO_ASCII = str.maketrans(
    {
        **dict.fromkeys(_HYPHEN_AND_DASH_VARIANTS, "-"),
        **dict.fromkeys(_CURLY_SINGLE_QUOTES, "'"),
        **dict.fromkeys(_CURLY_DOUBLE_QUOTES, '"'),
    }
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    without_articles = _SQUAD_ARTICLES_RE.sub(" ", lowered)
    without_punctuation = without_articles.translate(_SQUAD_PUNCTUATION_TABLE)
    return " ".join(without_punctuation.split())


def token_f1(prediction: str, gold: str) -> dict[str, float]:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        score = 1.0 if prediction_tokens == gold_tokens else 0.0
        return {"precision": score, "recall": score, "f1": score}
    multiset_overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if multiset_overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = multiset_overlap / len(prediction_tokens)
    recall = multiset_overlap / len(gold_tokens)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall),
    }


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(gold)


def substring_match(prediction: str, gold: str) -> bool:
    normalized_gold = normalize_answer(gold)
    if not normalized_gold:
        return exact_match(prediction, gold)
    return normalized_gold in normalize_answer(prediction)


def predicted_insufficient(prediction: str) -> bool:
    return normalize_answer(prediction).startswith(NORMALIZED_ABSTENTION_PREFIXES)


def answer_metrics(prediction: str, gold_answers: Sequence[str] | str) -> dict[str, Any]:
    references = (gold_answers,) if isinstance(gold_answers, str) else gold_answers
    golds = [str(gold) for gold in references]
    if not golds:
        raise ValueError("answer_metrics requires at least one gold answer.")
    best_scores = max(
        (token_f1(prediction, gold) for gold in golds), key=lambda scores: scores["f1"]
    )
    return {
        "exact_match": any(exact_match(prediction, gold) for gold in golds),
        "f1": best_scores["f1"],
        "precision": best_scores["precision"],
        "recall": best_scores["recall"],
        "substring_match": any(substring_match(prediction, gold) for gold in golds),
        "predicted_insufficient": predicted_insufficient(prediction),
    }


def normalize_passage_text(text: str) -> str:
    compatibility_form = unicodedata.normalize("NFKC", str(text))
    with_ascii_punctuation = compatibility_form.translate(_FACT_PUNCTUATION_TO_ASCII)
    return _WHITESPACE_RE.sub(" ", with_ascii_punctuation).casefold().strip()


def retrieval_metrics_for_query(
    query: RagQuery,
    hits: Sequence[SearchHit],
    chunk_text_by_id: Mapping[str, str],
) -> dict[str, Any] | None:
    if not query.evidence:
        return None

    evidence_doc_ids = {ref.doc_id for ref in query.evidence}
    hits_by_doc: dict[str, list[SearchHit]] = {}
    first_evidence_rank: int | None = None
    for hit in hits:
        doc_id = hit.payload.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"Retrieved chunk {hit.id} has no doc_id payload.")
        hits_by_doc.setdefault(doc_id, []).append(hit)
        if first_evidence_rank is None and doc_id in evidence_doc_ids:
            first_evidence_rank = hit.rank

    retrieved_evidence_doc_ids = evidence_doc_ids & set(hits_by_doc)
    retrieved_fact_count = sum(
        1
        for ref in query.evidence
        if _fact_in_retrieved_chunks(ref.fact, hits_by_doc.get(ref.doc_id, ()), chunk_text_by_id)
    )
    return {
        "evidence_doc_count": len(evidence_doc_ids),
        "retrieved_chunk_count": len(hits),
        "retrieved_doc_count": len(hits_by_doc),
        "evidence_recall_at_k": len(retrieved_evidence_doc_ids) / len(evidence_doc_ids),
        "evidence_full_recall_at_k": 1.0 if retrieved_evidence_doc_ids == evidence_doc_ids else 0.0,
        "evidence_hit_any_at_k": 1.0 if retrieved_evidence_doc_ids else 0.0,
        "doc_mrr_at_k": 0.0 if first_evidence_rank is None else 1.0 / first_evidence_rank,
        "fact_recall_at_k": retrieved_fact_count / len(query.evidence),
    }


def _fact_in_retrieved_chunks(
    fact: str,
    doc_hits: Sequence[SearchHit],
    chunk_text_by_id: Mapping[str, str],
) -> bool:
    if not doc_hits:
        return False
    hits_in_document_order = sorted(doc_hits, key=_hit_chunk_index)
    retrieved_text = "".join(
        chunk_text_by_id.get(str(hit.id), "") for hit in hits_in_document_order
    )
    normalized_fact = normalize_passage_text(fact)
    if not normalized_fact:
        return False
    return normalized_fact in normalize_passage_text(retrieved_text)


def _hit_chunk_index(hit: SearchHit) -> int:
    chunk_index = hit.payload.get("chunk_index")
    try:
        return int(chunk_index)
    except (TypeError, ValueError):
        return 0
