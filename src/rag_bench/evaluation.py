from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from locomo_jasper_bench.clients import ChatResult
from locomo_jasper_bench.vector_types import RetrievalMetrics, SearchHit

from .config import RagBenchConfig
from .data_types import RagDocument, RagQuery
from .metrics import answer_metrics


def final_answer_text(content: str) -> str:
    text = str(content).strip()
    if "ANSWER:" in text:
        return text.rsplit("ANSWER:", 1)[-1].strip()
    return text


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
    metrics: dict[str, Any] = {**answer.metrics}
    if answer.ttft_ms is not None:
        metrics["time_to_first_token_ms"] = answer.ttft_ms
    if retrieval_metrics is not None:
        metrics.update(
            {
                "query_embedding_time_ms": retrieval_metrics.embedding_time_ms,
                "vector_db_query_time_ms": retrieval_metrics.search_time_ms,
                "query_retrieval_time_ms": retrieval_metrics.total_time_ms,
            }
        )
        if retrieval_metrics.vector_backend is not None:
            metrics["resolved_vector_backend"] = retrieval_metrics.vector_backend
        if retrieval_metrics.jasper_effective_beam_width is not None:
            metrics["jasper_effective_beam_width"] = retrieval_metrics.jasper_effective_beam_width

    return {
        "run_id": config.run_id,
        "mode": config.result_mode(),
        "dataset": config.dataset_name,
        "query_id": query.query_id,
        "question_type": query.question_type,
        # Written as category too so type-agnostic tooling keeps working.
        "category": query.question_type,
        "question": query.question,
        "gold_answer": query.gold_answer,
        "gold_answers": list(query.gold_answers),
        "predicted_answer": answer.content,
        "evidence": [
            {"doc_id": ref.doc_id, "title": ref.title, "url": ref.url, "fact": ref.fact}
            for ref in query.evidence
        ],
        "retrieved_chunks": [
            {
                "chunk_id": str(hit.id),
                "doc_id": hit.payload.get("doc_id"),
                "chunk_index": hit.payload.get("chunk_index"),
                "rank": hit.rank,
                "score": hit.score,
                "distance": hit.distance,
                "title": _doc_title(docs_by_id, hit.payload.get("doc_id")),
            }
            for hit in hits
        ],
        "answer_metrics": answer_metrics(answer.content, query.gold_answers),
        "retrieval": retrieval_quality,
        "judge": judge_payload,
        "metrics": metrics,
    }


def _doc_title(docs_by_id: Mapping[str, RagDocument], doc_id: Any) -> str | None:
    if not isinstance(doc_id, str):
        return None
    doc = docs_by_id.get(doc_id)
    return doc.title if doc is not None else None
