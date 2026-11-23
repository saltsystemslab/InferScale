from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.files import JsonlWriter
from benchmarks.common.judge import judge_label
from benchmarks.common.vector_types import SearchHit
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.data import ConversationSample, QuestionAnswer, load_locomo
from benchmarks.memory.evaluation import answer_question
from benchmarks.memory.mem0.fact_catalog import fact_catalog_hits
from benchmarks.memory.mem0.memory_builder import SampleMemoryBuilder
from benchmarks.memory.protocol import SCORED_CATEGORIES
from benchmarks.memory.runtime_clients import (
    KVPrecomputeClient,
    MemoryWarmupClient,
    RuntimeClients,
)
from benchmarks.memory.setup_metrics import (
    empty_setup_row,
    record_kv_precompute,
    record_memory_warmup,
    setup_total_ms,
)

PREDICTIONS_FILE = "predictions.jsonl"


@dataclass(slots=True)
class PreparedSample:
    index: int
    sample: ConversationSample
    questions: list[QuestionAnswer]
    fact_catalog: list[SearchHit]
    setup_row: dict[str, Any]


@dataclass(slots=True)
class PredictionResult:
    records: list[dict[str, Any]]
    sample_setup_metrics: list[dict[str, Any]]


def run_prediction_mode(config: MemoryRunConfig, clients: RuntimeClients) -> PredictionResult:
    logger.info("Loading LoCoMo dataset from {}", config.dataset_path)
    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    planned_questions = planned_question_count(samples, config.max_questions)
    logger.info(
        "Loaded {} samples for answer_backend={}; planned_questions={} max_samples={} "
        "max_questions={} context_window={}",
        len(samples),
        config.answer_backend,
        planned_questions,
        config.max_samples,
        config.max_questions,
        config.context_window,
    )
    memory_builder = SampleMemoryBuilder(config)
    prepared_samples = _prepare_samples(config, clients, memory_builder, samples)
    logger.info(
        "Prepared {} samples and {} questions before vLLM startup",
        len(prepared_samples),
        sum(len(prepared.questions) for prepared in prepared_samples),
    )
    predictions_path = config.run_dir / PREDICTIONS_FILE
    if not prepared_samples:
        _write_empty_predictions_file(predictions_path)
        logger.info("Wrote 0 prediction records to {}", predictions_path)
        return PredictionResult(records=[], sample_setup_metrics=[])

    clients.answer_client.start_llm()
    progress = _Progress(planned_questions, config.log_every)
    records: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    with JsonlWriter(predictions_path) as writer:
        for prepared in prepared_samples:
            setup_row, sample_records = _answer_prepared_sample(
                config, clients, memory_builder, prepared, len(samples), writer, progress
            )
            setup_rows.append(setup_row)
            records.extend(sample_records)
    logger.info("Wrote {} prediction records to {}", len(records), predictions_path)
    return PredictionResult(records=records, sample_setup_metrics=setup_rows)


def planned_question_count(samples: list[ConversationSample], max_questions: int | None) -> int:
    total = sum(len(scored_questions(sample.qa)) for sample in samples)
    return total if max_questions is None else min(total, max_questions)


def scored_questions(questions: list[QuestionAnswer]) -> list[QuestionAnswer]:
    return [qa for qa in questions if str(qa.category).strip() in SCORED_CATEGORIES]


def should_log_progress(index: int, total: int, interval: int) -> bool:
    if interval <= 0 or total <= 0:
        return False
    return index == 1 or index == total or index % interval == 0


def _write_empty_predictions_file(predictions_path: Path) -> None:
    with JsonlWriter(predictions_path):
        pass


def _prepare_samples(
    config: MemoryRunConfig,
    clients: RuntimeClients,
    memory_builder: SampleMemoryBuilder,
    samples: list[ConversationSample],
) -> list[PreparedSample]:
    prepared_samples: list[PreparedSample] = []
    remaining_questions = config.max_questions
    for sample_number, sample in enumerate(samples, start=1):
        if remaining_questions is not None and remaining_questions <= 0:
            break
        questions = scored_questions(sample.qa)
        if remaining_questions is not None:
            questions = questions[:remaining_questions]
            remaining_questions -= len(questions)
        if not questions:
            continue
        logger.info(
            "Sample {}/{} sample_id={} turns={} questions={} preparing",
            sample_number,
            len(samples),
            sample.sample_id,
            len(sample.turns),
            len(questions),
        )
        fact_catalog = fact_catalog_hits(memory_builder.load_fact_catalog(sample))
        setup_row = empty_setup_row(config, sample, len(questions))
        if isinstance(clients.answer_client, KVPrecomputeClient):
            kv_metrics = clients.answer_client.precompute_sample_cache(sample, fact_catalog) or {}
            record_kv_precompute(setup_row, kv_metrics)
        prepared_samples.append(
            PreparedSample(
                index=sample_number,
                sample=sample,
                questions=questions,
                fact_catalog=fact_catalog,
                setup_row=setup_row,
            )
        )
    return prepared_samples


def _answer_prepared_sample(
    config: MemoryRunConfig,
    clients: RuntimeClients,
    memory_builder: SampleMemoryBuilder,
    prepared: PreparedSample,
    sample_count: int,
    writer: JsonlWriter,
    progress: _Progress,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample = prepared.sample
    retriever, memory_metrics = memory_builder.build_retriever_with_metrics(sample)
    setup_row = {**prepared.setup_row, **memory_metrics}
    records: list[dict[str, Any]] = []
    try:
        prepare_started = time.perf_counter()
        clients.answer_client.prepare_sample(sample)
        if isinstance(clients.answer_client, MemoryWarmupClient):
            warmup_metrics = clients.answer_client.warm_sample_memory(sample, prepared.fact_catalog)
            record_memory_warmup(setup_row, warmup_metrics or {})
        setup_row["answer_prepare_sample_time_ms"] = (time.perf_counter() - prepare_started) * 1000
        setup_row["sample_setup_time_ms"] = setup_total_ms(setup_row)
        for qa in prepared.questions:
            progress.log_start(sample, qa)
            query_started_at = time.perf_counter()
            hits, retrieval_metrics = retriever.search(qa.question, top_k=config.top_k)
            record = answer_question(
                config,
                clients,
                sample,
                qa,
                hits,
                retrieval_metrics=retrieval_metrics,
                ttft_started_at=time.perf_counter(),
                query_started_at=query_started_at,
            )
            writer.write(record)
            records.append(record)
            progress.log_finish(sample, qa, record)
        logger.info(
            "Sample {}/{} sample_id={} finished", prepared.index, sample_count, sample.sample_id
        )
    finally:
        retriever.log_cache_stats()
        retriever.close()
        clients.answer_client.close_sample()
    return setup_row, records


class _Progress:
    def __init__(self, planned_questions: int, log_interval: int) -> None:
        self.planned_questions = planned_questions
        self.log_interval = log_interval
        self.completed_questions = 0

    def log_start(self, sample: ConversationSample, qa: QuestionAnswer) -> None:
        question_number = self.completed_questions + 1
        if should_log_progress(question_number, self.planned_questions, self.log_interval):
            logger.info(
                "Question {}/{} starting sample_id={} question_id={} category={}",
                question_number,
                self.planned_questions,
                sample.sample_id,
                qa.question_id,
                qa.category,
            )

    def log_finish(
        self, sample: ConversationSample, qa: QuestionAnswer, record: dict[str, Any]
    ) -> None:
        self.completed_questions += 1
        if should_log_progress(self.completed_questions, self.planned_questions, self.log_interval):
            logger.info(
                "Question {}/{} finished sample_id={} question_id={} judge={}",
                self.completed_questions,
                self.planned_questions,
                sample.sample_id,
                qa.question_id,
                judge_label(record.get("judge", {}).get("correct")),
            )
