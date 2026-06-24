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
from .retrieval.memory_builder import SampleMemoryBuilder
from .results import JsonlWriter
from .vector_types import SearchHit


@dataclass(slots=True)
class PreparedQuestion:
    qa: QuestionAnswer
    hits: list[SearchHit]


@dataclass(slots=True)
class PreparedSample:
    index: int
    sample: ConversationSample
    questions: list[PreparedQuestion]


def run_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
    return run_kv_prediction_mode(config, clients)


def run_kv_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
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

    for sample_index, sample in enumerate(samples, start=1):
        if remaining_questions is not None and remaining_questions <= 0:
            break

        sample_questions = sample.qa
        if remaining_questions is not None:
            sample_questions = sample_questions[:remaining_questions]
        if not sample_questions:
            continue

        logger.info(
            "KV sample {}/{} sample_id={} turns={} questions={} retrieval starting",
            sample_index,
            len(samples),
            sample.sample_id,
            len(sample.turns),
            len(sample_questions),
        )

        memory = memory_builder.build(sample)
        prepared_questions: list[PreparedQuestion] = []
        try:
            for qa in sample_questions:
                hits = question_evaluator.search_mem0_memory(memory, qa.question)
                prepared_questions.append(PreparedQuestion(qa=qa, hits=hits))
        finally:
            memory_builder.log_embedding_cache_stats(memory, sample.sample_id)
            memory_builder.close(memory)

        if remaining_questions is not None:
            remaining_questions -= len(sample_questions)

        if not prepared_questions:
            continue

        logger.info(
            "KV sample {}/{} sample_id={} retrieval complete; vector indexes closed before encoder/vLLM load",
            sample_index,
            len(samples),
            sample.sample_id,
        )
        prepared_samples.append(PreparedSample(index=sample_index, sample=sample, questions=prepared_questions))
        if active_sample_gpu_cache:
            precompute_sample_cache(sample)

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
        return all_records

    start_llm()

    with JsonlWriter(output_path) as writer:
        for prepared in prepared_samples:
            sample = prepared.sample
            prepare_sample(sample, [(question.qa, question.hits) for question in prepared.questions])
            try:
                for question in prepared.questions:
                    qa = question.qa
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
                    record = question_evaluator.answer_from_hits(
                        sample,
                        qa,
                        question.hits,
                        ttft_started_at=time.perf_counter(),
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
                close_sample()

    logger.info("Wrote {} prepared vLLM prediction records to {}", len(all_records), output_path)
    return all_records


def planned_question_count(samples: list[ConversationSample], max_questions: int | None) -> int:
    total = sum(len(sample.qa) for sample in samples)
    if max_questions is None:
        return total
    return min(total, max_questions)


def should_log_progress(index: int, total: int, interval: int) -> bool:
    if interval <= 0 or total <= 0:
        return False
    return index == 1 or index == total or index % interval == 0
