from __future__ import annotations

import hashlib
from typing import Any

from .clients_factory import RuntimeClients
from .config import BenchmarkConfig
from .data import ConversationSample, QuestionAnswer
from .judging import judge_qa, skipped_judge_payload
from .modes import result_mode
from .vector_types import RetrievalMetrics, SearchHit


class QuestionEvaluator:
    def __init__(self, config: BenchmarkConfig, clients: RuntimeClients) -> None:
        self.config = config
        self.clients = clients

    def answer_from_hits(
        self,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        *,
        retrieval_metrics: RetrievalMetrics | None = None,
        ttft_started_at: float | None = None,
        query_started_at: float | None = None,
    ) -> dict[str, Any]:
        kv_answer = getattr(self.clients.answer_client, "answer_with_retrieved_memory", None)
        if not callable(kv_answer):
            raise RuntimeError(f"{self.config.answer_backend} answer backend cannot answer with retrieved memory.")
        answer = kv_answer(
            sample=sample,
            qa=qa,
            hits=hits,
            max_tokens=self.config.max_answer_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            ttft_started_at=ttft_started_at,
            query_started_at=query_started_at,
        )
        answer.content = _final_answer_text(answer.content)
        return self.record_answer(
            sample,
            qa,
            hits,
            answer,
            retrieval_metrics=retrieval_metrics,
        )

    def record_answer(
        self,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        answer: Any,
        *,
        retrieval_metrics: RetrievalMetrics | None = None,
    ) -> dict[str, Any]:
        if self.config.skip_judge:
            judge_payload = skipped_judge_payload(self.config)
        else:
            if self.clients.judge_client is None:
                raise RuntimeError("Judge client is not configured. Use --skip-judge to write unjudged predictions.")
            judge_payload = judge_qa(
                self.config,
                self.clients.judge_client,
                qa,
                answer.content,
                evidence_context=_evidence_context(sample, qa) if self.config.with_evidence else "",
            )

        metrics: dict[str, Any] = {**getattr(answer, "metrics", {})}
        metrics["memory_retrieved_fact_ids"] = [hit.id for hit in hits]
        metrics["memory_retrieved_fact_text_hashes"] = [
            _memory_text_hash(hit) for hit in hits
        ]
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
            "run_id": self.config.run_id,
            "mode": result_mode(self.config),
            "sample_id": sample.sample_id,
            "question_id": qa.question_id,
            "category": qa.category,
            "question": qa.question,
            "gold_answer": qa.answer,
            "predicted_answer": answer.content,
            "evidence": qa.evidence,
            "evidence_context": _evidence_context(sample, qa),
            "retrieved_memories": [
                {
                    "id": hit.id,
                    "rank": hit.rank,
                    "score": hit.score,
                    "distance": hit.distance,
                    "memory": hit.payload.get("memory") or hit.payload.get("text") or "",
                    "created_at": hit.payload.get("created_at"),
                    "source_session_id": _payload_value(hit.payload, "source_session_id", "session_id"),
                    "source_session_index": _payload_value(
                        hit.payload,
                        "source_session_index",
                        "session_index",
                    ),
                    "source_turn_id": _payload_value(hit.payload, "source_turn_id", "turn_id"),
                    "metadata": hit.payload.get("metadata", {}),
                }
                for hit in hits
            ],
            "judge": judge_payload,
            "metrics": metrics,
        }

def _final_answer_text(content: str) -> str:
    text = str(content).strip()
    if "ANSWER:" in text:
        return text.rsplit("ANSWER:", 1)[-1].strip()
    return text


def _memory_text_hash(hit: SearchHit) -> str:
    text = str(hit.payload.get("memory") or hit.payload.get("text") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_value(payload: dict[str, Any], primary: str, legacy: str) -> Any:
    value = payload.get(primary)
    if value is not None:
        return value
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get(primary)
        if value is not None:
            return value
        return metadata.get(legacy)
    return payload.get(legacy)


def _evidence_context(sample: ConversationSample, qa: QuestionAnswer) -> str:
    evidence = qa.evidence
    if isinstance(evidence, str):
        evidence_ids = [evidence]
    elif isinstance(evidence, list):
        evidence_ids = [str(item) for item in evidence]
    else:
        return ""
    requested = set(evidence_ids)
    if not requested:
        return ""
    by_id: dict[str, str] = {}
    for turn in sample.turns:
        raw = turn.raw if isinstance(turn.raw, dict) else {}
        dialogue_id = str(raw.get("dia_id") or "")
        if not dialogue_id or dialogue_id not in requested:
            continue
        date_suffix = f", said on {turn.timestamp}" if turn.timestamp else ""
        by_id[dialogue_id] = (
            f'[{dialogue_id}{date_suffix}] {turn.speaker}: "{turn.text}"'
        )
    return "\n".join(by_id[evidence_id] for evidence_id in evidence_ids if evidence_id in by_id)
