from __future__ import annotations

from typing import Any

from inferscale.v1.serving.vllm import GenerationOutput
from inferscale.v1.types import RetrievalMetrics


def generation_latency_metrics(
    output: GenerationOutput,
    *,
    request_started: float,
    query_started_at: float | None,
    engine_ttft_key: str,
    excluded_ms: float = 0.0,
) -> dict[str, float]:
    generate_started = output.started_at
    finished = output.finished_at
    ttft_ms = output.ttft_ms
    metrics = {
        "answer_generate_time_ms": (finished - generate_started) * 1000,
        "answer_total_time_ms": _elapsed_ms(request_started, finished, excluded_ms),
        engine_ttft_key: ttft_ms,
        "answer_time_to_first_token_ms": (
            _elapsed_ms(request_started, generate_started, excluded_ms) + ttft_ms
        ),
    }
    if query_started_at is not None:
        metrics["query_to_answer_ms"] = _elapsed_ms(query_started_at, finished, excluded_ms)
        metrics["query_to_first_token_ms"] = (
            _elapsed_ms(query_started_at, generate_started, excluded_ms) + ttft_ms
        )
    return metrics


def retrieval_latency_metrics(retrieval: RetrievalMetrics) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "query_embedding_time_ms": retrieval.embedding_time_ms,
        "vector_db_query_time_ms": retrieval.search_time_ms,
        "query_retrieval_time_ms": retrieval.total_time_ms,
    }
    if retrieval.vector_backend is not None:
        metrics["resolved_vector_backend"] = retrieval.vector_backend
    if retrieval.jasper_effective_beam_width is not None:
        metrics["jasper_effective_beam_width"] = retrieval.jasper_effective_beam_width
    return metrics


def _elapsed_ms(started: float, finished: float, excluded_ms: float) -> float:
    return max(0.0, (finished - started) * 1000 - excluded_ms)
