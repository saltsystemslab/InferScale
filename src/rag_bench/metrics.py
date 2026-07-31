from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from locomo_jasper_bench.vector_types import SearchHit

from .data_types import RagQuery

# SQuAD-style answer normalization and token F1, following the legacy scorer in
# ai-memory-code/mem0_locomo/score.py (lowercase, strip articles and
# punctuation, collapse whitespace, multiset token overlap).
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Normalized prefixes that count as abstention across datasets: MultiHop-RAG
# uses "Insufficient information", QASPER uses "Unanswerable".
ABSTENTION_ANSWER_PREFIXES = ("insufficient information", "unanswerable")

# Unicode punctuation that appears verbatim in MultiHop-RAG evidence facts,
# mapped to ascii before substring matching so chunk decodes and facts compare
# on equal footing. Written as escapes on purpose: hyphen and dash variants
# (u2010-u2015), curly single quotes (u2018/u2019/u201a), curly double quotes
# (u201c/u201d/u201e), narrow no-break space (u202f), ellipsis (u2026).
_PASSAGE_CHAR_MAP = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u202f": " ",
        "\u2026": "...",
    }
)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    without_articles = _ARTICLES_RE.sub(" ", lowered)
    without_punct = without_articles.translate(_PUNCT_TABLE)
    return " ".join(without_punct.split())


def token_f1(prediction: str, gold: str) -> dict[str, float]:
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        value = 1.0 if prediction_tokens == gold_tokens else 0.0
        return {"precision": value, "recall": value, "f1": value}
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
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
    return normalize_answer(prediction).startswith(ABSTENTION_ANSWER_PREFIXES)


def answer_metrics(prediction: str, gold_answers: Sequence[str] | str) -> dict[str, Any]:
    """String metrics against one or more reference answers.

    Follows the official QASPER convention: F1 (with its precision/recall) is
    the best over references, and exact/substring match count if any reference
    matches. Single-reference datasets pass a one-element sequence.
    """
    if isinstance(gold_answers, str):
        gold_answers = (gold_answers,)
    golds = [str(gold) for gold in gold_answers]
    if not golds:
        raise ValueError("answer_metrics requires at least one gold answer.")
    best_f1 = max((token_f1(prediction, gold) for gold in golds), key=lambda item: item["f1"])
    return {
        "exact_match": any(exact_match(prediction, gold) for gold in golds),
        "f1": best_f1["f1"],
        "precision": best_f1["precision"],
        "recall": best_f1["recall"],
        "substring_match": any(substring_match(prediction, gold) for gold in golds),
        "predicted_insufficient": predicted_insufficient(prediction),
    }


def normalize_passage_text(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.translate(_PASSAGE_CHAR_MAP)
    return _WHITESPACE_RE.sub(" ", normalized).casefold().strip()


def retrieval_metrics_for_query(
    query: RagQuery,
    hits: Sequence[SearchHit],
    chunk_text_by_id: Mapping[str, str],
) -> dict[str, Any] | None:
    """Doc- and fact-level retrieval quality against the gold evidence.

    Returns None for null queries, which have no evidence and are excluded
    from retrieval aggregates by design.
    """
    if not query.evidence:
        return None

    evidence_doc_ids: list[str] = list(dict.fromkeys(ref.doc_id for ref in query.evidence))
    retrieved_doc_ids: list[str] = []
    chunks_by_doc: dict[str, list[SearchHit]] = {}
    first_evidence_rank: int | None = None
    evidence_doc_set = set(evidence_doc_ids)
    for hit in hits:
        doc_id = hit.payload.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"Retrieved chunk {hit.id} has no doc_id payload.")
        if doc_id not in chunks_by_doc:
            retrieved_doc_ids.append(doc_id)
        chunks_by_doc.setdefault(doc_id, []).append(hit)
        if first_evidence_rank is None and doc_id in evidence_doc_set:
            first_evidence_rank = hit.rank

    covered_docs = evidence_doc_set & set(retrieved_doc_ids)
    fact_covered = sum(
        1
        for ref in query.evidence
        if _fact_in_retrieved_chunks(ref.fact, chunks_by_doc.get(ref.doc_id, ()), chunk_text_by_id)
    )
    return {
        "evidence_doc_count": len(evidence_doc_ids),
        "retrieved_chunk_count": len(hits),
        "retrieved_doc_count": len(retrieved_doc_ids),
        "evidence_recall_at_k": len(covered_docs) / len(evidence_doc_ids),
        "evidence_full_recall_at_k": 1.0 if covered_docs == evidence_doc_set else 0.0,
        "evidence_hit_any_at_k": 1.0 if covered_docs else 0.0,
        "doc_mrr_at_k": 0.0 if first_evidence_rank is None else 1.0 / first_evidence_rank,
        "fact_recall_at_k": fact_covered / len(query.evidence),
    }


def _fact_in_retrieved_chunks(
    fact: str,
    doc_hits: Sequence[SearchHit],
    chunk_text_by_id: Mapping[str, str],
) -> bool:
    if not doc_hits:
        return False
    ordered = sorted(doc_hits, key=_hit_chunk_index)
    joined = "".join(chunk_text_by_id.get(str(hit.id), "") for hit in ordered)
    normalized_fact = normalize_passage_text(fact)
    if not normalized_fact:
        return False
    return normalized_fact in normalize_passage_text(joined)


def _hit_chunk_index(hit: SearchHit) -> int:
    value = hit.payload.get("chunk_index")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
