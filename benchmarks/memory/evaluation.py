"""Answer one question with the retrieved facts, judge it, and build its prediction record."""

from __future__ import annotations

from typing import Any

from benchmarks.common.answer_timing import retrieval_latency_metrics
from benchmarks.common.answers import final_answer_text
from benchmarks.common.clients import ChatResult
from benchmarks.common.judge import skipped_judge_payload
from benchmarks.common.vector_types import RetrievalMetrics, SearchHit
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, QuestionAnswer
from benchmarks.memory.facts import payload_value, text_hash
from benchmarks.memory.judging import judge_metadata, judge_qa
from benchmarks.memory.modes import result_mode
from benchmarks.memory.runtime_clients import RuntimeClients


def answer_question(
    config: MemoryRunConfig,
    clients: RuntimeClients,
    sample: ConversationSample,
    qa: QuestionAnswer,
    hits: list[SearchHit],
    *,
    retrieval_metrics: RetrievalMetrics | None = None,
    ttft_started_at: float | None = None,
    query_started_at: float | None = None,
) -> dict[str, Any]:
    answer = clients.answer_client.answer_with_retrieved_memory(
        sample=sample,
        qa=qa,
        hits=hits,
        max_tokens=config.max_answer_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        ttft_started_at=ttft_started_at,
        query_started_at=query_started_at,
    )
    answer.content = final_answer_text(answer.content)
    evidence = evidence_context(sample, qa)
    judge = _judge_answer(config, clients, qa, answer.content, evidence)
    return build_prediction_record(
        config,
        sample,
        qa,
        hits,
        answer,
        judge_payload=judge,
        evidence=evidence,
        retrieval_metrics=retrieval_metrics,
    )


def build_prediction_record(
    config: MemoryRunConfig,
    sample: ConversationSample,
    qa: QuestionAnswer,
    hits: list[SearchHit],
    answer: ChatResult,
    *,
    judge_payload: dict[str, Any],
    evidence: str,
    retrieval_metrics: RetrievalMetrics | None = None,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "mode": result_mode(config),
        "sample_id": sample.sample_id,
        "question_id": qa.question_id,
        "category": qa.category,
        "question": qa.question,
        "gold_answer": qa.answer,
        "predicted_answer": answer.content,
        "evidence": qa.evidence,
        "evidence_context": evidence,
        "retrieved_memories": [_retrieved_memory_row(hit) for hit in hits],
        "judge": judge_payload,
        "metrics": _answer_metrics(answer, hits, retrieval_metrics),
    }


def evidence_context(sample: ConversationSample, qa: QuestionAnswer) -> str:
    """The evidence turns of a question, rendered the way the upstream judge prompt shows them."""
    evidence = qa.evidence
    if isinstance(evidence, str):
        evidence_ids = [evidence]
    elif isinstance(evidence, list):
        evidence_ids = [str(item) for item in evidence]
    else:
        return ""
    requested = set(evidence_ids)
    by_id: dict[str, str] = {}
    for turn in sample.turns:
        raw = turn.raw if isinstance(turn.raw, dict) else {}
        dialogue_id = str(raw.get("dia_id") or "")
        if dialogue_id in requested:
            date_suffix = f", said on {turn.timestamp}" if turn.timestamp else ""
            by_id[dialogue_id] = f'[{dialogue_id}{date_suffix}] {turn.speaker}: "{turn.text}"'
    return "\n".join(by_id[evidence_id] for evidence_id in evidence_ids if evidence_id in by_id)


def _judge_answer(
    config: MemoryRunConfig,
    clients: RuntimeClients,
    qa: QuestionAnswer,
    predicted_answer: str,
    evidence: str,
) -> dict[str, Any]:
    if config.skip_judge:
        return skipped_judge_payload(judge_metadata(config))
    if clients.judge_client is None:
        raise RuntimeError(
            "Judge client is not configured. Set skip_judge=true in JSON to write "
            "unjudged predictions."
        )
    return judge_qa(
        config,
        clients.judge_client,
        qa,
        predicted_answer,
        evidence_context=evidence if config.with_evidence else "",
    )


def _retrieved_memory_row(hit: SearchHit) -> dict[str, Any]:
    return {
        "id": hit.id,
        "rank": hit.rank,
        "score": hit.score,
        "distance": hit.distance,
        "memory": _memory_text(hit),
        "created_at": hit.payload.get("created_at"),
        "source_session_id": payload_value(hit.payload, ("source_session_id", "session_id")),
        "source_session_index": payload_value(
            hit.payload, ("source_session_index", "session_index")
        ),
        "source_turn_id": payload_value(hit.payload, ("source_turn_id", "turn_id")),
        "metadata": hit.payload.get("metadata", {}),
    }


def _answer_metrics(
    answer: ChatResult, hits: list[SearchHit], retrieval_metrics: RetrievalMetrics | None
) -> dict[str, Any]:
    metrics: dict[str, Any] = {**answer.metrics}
    metrics["memory_retrieved_fact_ids"] = [hit.id for hit in hits]
    metrics["memory_retrieved_fact_text_hashes"] = [text_hash(_memory_text(hit)) for hit in hits]
    if answer.ttft_ms is not None:
        metrics["time_to_first_token_ms"] = answer.ttft_ms
    if retrieval_metrics is not None:
        metrics.update(retrieval_latency_metrics(retrieval_metrics))
    return metrics


def _memory_text(hit: SearchHit) -> str:
    return str(hit.payload.get("memory") or hit.payload.get("text") or "")
