from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .clients_factory import RuntimeClients
from .config import BenchmarkConfig
from .data import ConversationSample, QuestionAnswer, load_locomo
from .evaluation import QuestionEvaluator
from .judging import judge_label
from .modes import result_mode
from .retrieval.memory_builder import SampleMemoryBuilder
from .results import JsonlWriter


@dataclass(slots=True)
class PreparedSample:
    index: int
    sample: ConversationSample
    questions: list[QuestionAnswer]
    setup_row: dict[str, Any]


@dataclass(slots=True)
class PredictionResult:
    records: list[dict[str, Any]]
    sample_setup_metrics: list[dict[str, Any]]


def run_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> PredictionResult:
    return run_kv_prediction_mode(config, clients)


def run_kv_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> PredictionResult:
    logger.info("Loading LoCoMo dataset from {}", config.dataset_path)
    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    planned_questions = planned_question_count(samples, config.max_questions)
    logger.info(
        "Loaded {} samples for prepared vLLM backend={}; planned_questions={} max_samples={} max_questions={} context_window={}",
        len(samples),
        config.answer_backend,
        planned_questions,
        config.max_samples,
        config.max_questions,
        config.context_window,
    )

    prepare_sample = getattr(clients.answer_client, "prepare_sample", None)
    close_sample = getattr(clients.answer_client, "close_sample", None)
    start_llm = getattr(clients.answer_client, "start_llm", None)
    if not callable(close_sample) or not callable(prepare_sample) or not callable(start_llm):
        raise RuntimeError(f"{config.answer_backend} answer backend does not expose sample preparation methods.")
    precompute_sample_cache = getattr(clients.answer_client, "precompute_sample_cache", None)
    active_sample_gpu_cache = callable(precompute_sample_cache)

    output_path = config.run_dir / "predictions.jsonl"
    all_records: list[dict[str, Any]] = []
    remaining_questions = config.max_questions
    completed_questions = 0
    memory_builder = SampleMemoryBuilder(config)
    question_evaluator = QuestionEvaluator(config, clients)
    prepared_samples: list[PreparedSample] = []
    sample_setup_rows: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples, start=1):
        if remaining_questions is not None and remaining_questions <= 0:
            break

        sample_questions = sample.qa
        if remaining_questions is not None:
            sample_questions = sample_questions[:remaining_questions]
        if not sample_questions:
            continue

        logger.info(
            "KV sample {}/{} sample_id={} turns={} questions={} preparation starting",
            sample_index,
            len(samples),
            sample.sample_id,
            len(sample.turns),
            len(sample_questions),
        )

        if remaining_questions is not None:
            remaining_questions -= len(sample_questions)

        logger.info(
            "KV sample {}/{} sample_id={} selected; query retrieval deferred until answer timing",
            sample_index,
            len(samples),
            sample.sample_id,
        )
        setup_row = _base_sample_setup_row(config, sample, len(sample_questions))
        if active_sample_gpu_cache:
            kv_metrics = precompute_sample_cache(sample) or {}
            setup_row["kv_precompute_time_ms"] = _number(kv_metrics.get("kv_precompute_time_ms"))
        prepared_samples.append(
            PreparedSample(
                index=sample_index,
                sample=sample,
                questions=list(sample_questions),
                setup_row=setup_row,
            )
        )

    if active_sample_gpu_cache:
        logger.info(
            "Precomputed GPU-resident KV caches for {} samples and {} questions before one vLLM startup",
            len(prepared_samples),
            sum(len(prepared.questions) for prepared in prepared_samples),
        )
    else:
        logger.info(
            "Prepared {} samples and {} questions before vLLM startup",
            len(prepared_samples),
            sum(len(prepared.questions) for prepared in prepared_samples),
        )
    if not prepared_samples:
        with JsonlWriter(output_path):
            pass
        logger.info("Wrote 0 prepared vLLM prediction records to {}", output_path)
        return PredictionResult(records=all_records, sample_setup_metrics=sample_setup_rows)

    start_llm()

    with JsonlWriter(output_path) as writer:
        for prepared in prepared_samples:
            sample = prepared.sample
            setup_row = prepared.setup_row
            memory, memory_metrics = memory_builder.build_with_metrics(sample)
            setup_row.update(memory_metrics)
            try:
                prepare_started = time.perf_counter()
                prepare_sample(sample)
                setup_row["answer_prepare_sample_time_ms"] = (time.perf_counter() - prepare_started) * 1000
                setup_row["sample_setup_time_ms"] = _setup_total_ms(setup_row)
                sample_setup_rows.append(dict(setup_row))
                for qa in prepared.questions:
                    next_question = completed_questions + 1
                    if should_log_progress(next_question, planned_questions, config.log_every):
                        logger.info(
                            "KV question {}/{} starting sample_id={} question_id={} category={}",
                            next_question,
                            planned_questions,
                            sample.sample_id,
                            qa.question_id,
                            qa.category,
                        )
                    query_started_at = time.perf_counter()
                    hits, retrieval_metrics = question_evaluator.retrieve_mem0_memory(memory, qa.question)
                    record = question_evaluator.answer_from_hits(
                        sample,
                        qa,
                        hits,
                        retrieval_metrics=retrieval_metrics,
                        ttft_started_at=time.perf_counter(),
                        query_started_at=query_started_at,
                    )
                    writer.write(record)
                    all_records.append(record)
                    completed_questions += 1
                    if should_log_progress(completed_questions, planned_questions, config.log_every):
                        logger.info(
                            "KV question {}/{} finished sample_id={} question_id={} judge={}",
                            completed_questions,
                            planned_questions,
                            sample.sample_id,
                            qa.question_id,
                            judge_label(record.get("judge", {}).get("correct")),
                        )
                logger.info(
                    "KV sample {}/{} sample_id={} finished",
                    prepared.index,
                    len(samples),
                    sample.sample_id,
                )
            finally:
                memory_builder.log_embedding_cache_stats(memory, sample.sample_id)
                memory_builder.close(memory)
                close_sample()

    logger.info("Wrote {} prepared vLLM prediction records to {}", len(all_records), output_path)
    return PredictionResult(records=all_records, sample_setup_metrics=sample_setup_rows)


def _base_sample_setup_row(
    config: BenchmarkConfig,
    sample: ConversationSample,
    question_count: int,
) -> dict[str, Any]:
    return {
        "run_id": config.run_id,
        "mode": result_mode(config),
        "sample_id": sample.sample_id,
        "question_count": question_count,
        "turn_count": len(sample.turns),
        "vector_backend": config.vector_backend,
        "memory_create_time_ms": None,
        "embedding_memory_build_time_ms": None,
        "vector_index_build_time_ms": None,
        "memory_setup_time_ms": None,
        "kv_precompute_time_ms": None,
        "answer_prepare_sample_time_ms": None,
        "sample_setup_time_ms": None,
    }


def _setup_total_ms(row: dict[str, Any]) -> float:
    return sum(
        value
        for value in (
            _number(row.get("memory_setup_time_ms")),
            _number(row.get("kv_precompute_time_ms")),
            _number(row.get("answer_prepare_sample_time_ms")),
        )
        if value is not None
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def planned_question_count(samples: list[ConversationSample], max_questions: int | None) -> int:
    total = sum(len(sample.qa) for sample in samples)
    if max_questions is None:
        return total
    return min(total, max_questions)


def should_log_progress(index: int, total: int, interval: int) -> bool:
    if interval <= 0 or total <= 0:
        return False
    return index == 1 or index == total or index % interval == 0
