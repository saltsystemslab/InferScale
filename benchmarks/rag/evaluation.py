"""The prediction record of one RAG query: answer, verdict, string metrics, retrieved chunks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.common.answer_timing import retrieval_latency_metrics
from benchmarks.common.clients import ChatResult
from benchmarks.common.vector_types import RetrievalMetrics, SearchHit
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import EvidenceRef, RagDocument, RagQuery
from benchmarks.rag.metrics import answer_metrics


def build_query_record(
    config: RagBenchConfig,
    query: RagQuery,
    hits: list[SearchHit],
    answer: ChatResult,
    *,
    retrieval_metrics: RetrievalMetrics | None,
    retrieval_quality: dict[str, Any] | None,
    judge_payload: dict[str, Any],
    docs_by_id: Mapping[str, RagDocument],
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "mode": config.result_mode(),
        "dataset": config.dataset_name,
        "query_id": query.query_id,
        "question_type": query.question_type,
        # category mirrors question_type: per-category reports are shared with the memory benchmark.
        "category": query.question_type,
        "question": query.question,
        "gold_answer": query.gold_answer,
        "gold_answers": list(query.gold_answers),
        "predicted_answer": answer.content,
        "evidence": [_evidence_row(ref) for ref in query.evidence],
        "retrieved_chunks": [_retrieved_chunk_row(hit, docs_by_id) for hit in hits],
        "answer_metrics": answer_metrics(answer.content, query.gold_answers),
        "retrieval": retrieval_quality,
        "judge": judge_payload,
        "metrics": _timing_metrics(answer, retrieval_metrics),
    }


def _timing_metrics(answer: ChatResult, retrieval: RetrievalMetrics | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {**answer.metrics}
    if answer.ttft_ms is not None:
        metrics["time_to_first_token_ms"] = answer.ttft_ms
    if retrieval is not None:
        metrics.update(retrieval_latency_metrics(retrieval))
    return metrics


def _evidence_row(ref: EvidenceRef) -> dict[str, Any]:
    return {"doc_id": ref.doc_id, "title": ref.title, "url": ref.url, "fact": ref.fact}


def _retrieved_chunk_row(hit: SearchHit, docs_by_id: Mapping[str, RagDocument]) -> dict[str, Any]:
    doc_id = hit.payload.get("doc_id")
    doc = docs_by_id.get(doc_id) if isinstance(doc_id, str) else None
    return {
        "chunk_id": str(hit.id),
        "doc_id": doc_id,
        "chunk_index": hit.payload.get("chunk_index"),
        "rank": hit.rank,
        "score": hit.score,
        "distance": hit.distance,
        "title": doc.title if doc is not None else None,
    }
