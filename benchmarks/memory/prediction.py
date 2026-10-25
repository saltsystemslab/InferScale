"""The run stage: precompute each sample's KV, start vLLM once, then retrieve and answer.

Samples are prepared in two passes because the HF encoder and vLLM cannot
share the GPU: every sample's fact catalog is loaded and (for KV injection)
encoded first, then vLLM starts and each sample's live retriever answers its
questions. A backend that warms a cache from the catalog does so once vLLM
runs, as part of the sample's preparation time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    planned = planned_question_count(samples, config.max_questions)
    logger.info(
        "Loaded {} samples for answer_backend={}; planned_questions={} max_samples={} "
        "max_questions={} context_window={}",
        len(samples),
        config.answer_backend,
        planned,
        config.max_samples,
        config.max_questions,
        config.context_window,
    )
    builder = SampleMemoryBuilder(config)
    prepared = _prepare_samples(config, clients, builder, samples)
    logger.info(
        "Prepared {} samples and {} questions before vLLM startup",
        len(prepared),
        sum(len(item.questions) for item in prepared),
    )
    output_path = config.run_dir / PREDICTIONS_FILE
    if not prepared:
        with JsonlWriter(output_path):
            pass
        logger.info("Wrote 0 prediction records to {}", output_path)
        return PredictionResult(records=[], sample_setup_metrics=[])

    clients.answer_client.start_llm()
    progress = _Progress(planned, config.log_every)
    records: list[dict[str, Any]] = []
    setup_rows: list[dict[str, Any]] = []
    with JsonlWriter(output_path) as writer:
        for item in prepared:
            setup_row, sample_records = _answer_prepared_sample(
                config, clients, builder, item, len(samples), writer, progress
            )
            setup_rows.append(setup_row)
            records.extend(sample_records)
    logger.info("Wrote {} prediction records to {}", len(records), output_path)
    return PredictionResult(records=records, sample_setup_metrics=setup_rows)


def planned_question_count(samples: list[ConversationSample], max_questions: int | None) -> int:
    total = sum(len(scored_questions(sample.qa)) for sample in samples)
    return total if max_questions is None else min(total, max_questions)


def scored_questions(questions: list[QuestionAnswer]) -> list[QuestionAnswer]:
    """The questions the upstream scorer counts; adversarial questions are left out."""
    return [qa for qa in questions if str(qa.category).strip() in SCORED_CATEGORIES]


def should_log_progress(index: int, total: int, interval: int) -> bool:
    if interval <= 0 or total <= 0:
        return False
    return index == 1 or index == total or index % interval == 0


def _prepare_samples(
    config: MemoryRunConfig,
    clients: RuntimeClients,
    builder: SampleMemoryBuilder,
    samples: list[ConversationSample],
) -> list[PreparedSample]:
    """Load each sample's catalog and, for KV injection, encode it before vLLM starts."""
    prepared: list[PreparedSample] = []
    remaining = config.max_questions
    for index, sample in enumerate(samples, start=1):
        if remaining is not None and remaining <= 0:
            break
        questions = scored_questions(sample.qa)
        if remaining is not None:
            questions = questions[:remaining]
            remaining -= len(questions)
        if not questions:
            continue
        logger.info(
            "Sample {}/{} sample_id={} turns={} questions={} preparing",
            index,
            len(samples),
            sample.sample_id,
            len(sample.turns),
            len(questions),
        )
        facts = fact_catalog_hits(builder.load_fact_catalog(sample))
        setup_row = empty_setup_row(config, sample, len(questions))
        if isinstance(clients.answer_client, KVPrecomputeClient):
            kv_metrics = clients.answer_client.precompute_sample_cache(sample, facts) or {}
            record_kv_precompute(setup_row, kv_metrics)
        prepared.append(PreparedSample(index, sample, questions, facts, setup_row))
    return prepared


def _answer_prepared_sample(
    config: MemoryRunConfig,
    clients: RuntimeClients,
    builder: SampleMemoryBuilder,
    prepared: PreparedSample,
    sample_count: int,
    writer: JsonlWriter,
    progress: _Progress,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Answer one prepared sample's questions; returns its setup row and records."""
    sample = prepared.sample
    retriever, memory_metrics = builder.build_retriever_with_metrics(sample)
    setup_row = {**prepared.setup_row, **memory_metrics}
    records: list[dict[str, Any]] = []
    try:
        prepare_started = time.perf_counter()
        clients.answer_client.prepare_sample(sample)
        if isinstance(clients.answer_client, MemoryWarmupClient):
            warmup = clients.answer_client.warm_sample_memory(sample, prepared.fact_catalog)
            record_memory_warmup(setup_row, warmup or {})
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
    """Question-level progress logging at the configured interval."""

    def __init__(self, planned: int, interval: int) -> None:
        self.planned = planned
        self.interval = interval
        self.completed = 0

    def log_start(self, sample: ConversationSample, qa: QuestionAnswer) -> None:
        if should_log_progress(self.completed + 1, self.planned, self.interval):
            logger.info(
                "Question {}/{} starting sample_id={} question_id={} category={}",
                self.completed + 1,
                self.planned,
                sample.sample_id,
                qa.question_id,
                qa.category,
            )

    def log_finish(
        self, sample: ConversationSample, qa: QuestionAnswer, record: dict[str, Any]
    ) -> None:
        self.completed += 1
        if should_log_progress(self.completed, self.planned, self.interval):
            logger.info(
                "Question {}/{} finished sample_id={} question_id={} judge={}",
                self.completed,
                self.planned,
                sample.sample_id,
                qa.question_id,
                judge_label(record.get("judge", {}).get("correct")),
            )
