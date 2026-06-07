from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from loguru import logger

from .clients import (
    ChatClient,
    OpenAICompatibleChatClient,
)
from .config import BenchmarkConfig
from .data import ConversationSample, QuestionAnswer, load_locomo
from .memory_builder import SampleMemoryBuilder, embed_mem0_query
from .prompts import build_judge_messages, build_retrieval_answer_messages, parse_judge_response
from .results import JsonlWriter, summarize_records, write_json
from .search_results import mem0_results_to_search_hits
from .system import collect_system_metadata
from .vector_types import SearchMetrics


@dataclass(slots=True)
class RuntimeClients:
    answer_client: ChatClient
    judge_client: ChatClient


def run_benchmark(config: BenchmarkConfig, clients: RuntimeClients | None = None) -> dict[str, Any]:
    logger.info(
        "Starting benchmark run_id={} dataset={} results_dir={}",
        config.run_id,
        config.dataset_path,
        config.run_dir,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.json", config.to_jsonable())

    runtime_clients = clients or build_clients(config)
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / "system.json", system_metadata)

    records = _run_prediction_mode(config, runtime_clients)

    summary = summarize_records(
        records,
        run_id=config.run_id,
        mode="baseline",
        config=config.to_jsonable(),
        system_metadata=system_metadata,
    )
    write_json(config.run_dir / "summary.json", summary)
    logger.info(
        "Finished benchmark run_id={} questions={} judged={} accuracy={}",
        config.run_id,
        summary["question_count"],
        summary["judged_count"],
        _format_accuracy(summary.get("metrics", {}).get("accuracy")),
    )
    logger.info("Wrote results to {}", config.run_dir)
    return summary


def build_clients(config: BenchmarkConfig) -> RuntimeClients:
    logger.info(
        "Configuring clients llm={} judge={} vector_backend={}",
        config.llm_base_url,
        config.judge_base_url,
        config.vector_backend,
    )
    answer_client = OpenAICompatibleChatClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.model,
        stream=config.stream,
    )
    judge_client = OpenAICompatibleChatClient(
        base_url=config.judge_base_url,
        api_key=config.judge_api_key,
        model=config.judge_model,
        stream=config.stream,
    )
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client)


class QuestionEvaluator:
    def __init__(self, config: BenchmarkConfig, clients: RuntimeClients) -> None:
        self.config = config
        self.clients = clients

    def answer(self, sample: ConversationSample, qa: QuestionAnswer, memory: Any | None) -> dict[str, Any]:
        ttft_started_at = time.perf_counter()
        if memory is None:
            raise RuntimeError("Mem0 context requires a memory store.")
        raw_results = self._search_mem0_memory(memory, qa.question)
        hits = mem0_results_to_search_hits(raw_results)
        answer_messages = build_retrieval_answer_messages(sample, qa, hits)
        store_metrics = self._mem0_store_search_metrics(memory)
        vector_search_ms = store_metrics.search_time_ms
        answer = self.clients.answer_client.chat(
            answer_messages,
            max_tokens=self.config.max_answer_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            ttft_started_at=ttft_started_at,
        )
        judge_messages = build_judge_messages(qa, answer.content)
        judge = self.clients.judge_client.chat(
            judge_messages,
            max_tokens=self.config.max_judge_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        correct, reason = parse_judge_response(judge.content)
        judge_payload = {"correct": correct, "reason": reason, "raw": judge.content}

        return {
            "run_id": self.config.run_id,
            "mode": "baseline",
            "sample_id": sample.sample_id,
            "question_id": qa.question_id,
            "category": qa.category,
            "question": qa.question,
            "gold_answer": qa.answer,
            "predicted_answer": answer.content,
            "evidence": qa.evidence,
            "retrieved_memories": [
                {
                    "id": hit.id,
                    "rank": hit.rank,
                    "score": hit.score,
                    "distance": hit.distance,
                    "memory": hit.payload.get("memory") or hit.payload.get("text") or "",
                    "metadata": hit.payload.get("metadata", {}),
                }
                for hit in hits
            ],
            "judge": judge_payload,
            "metrics": {
                "time_to_first_token_ms": answer.ttft_ms,
                "vector_db_query_time_ms": vector_search_ms,
            },
        }

    def _mem0_store_search_metrics(self, memory: Any) -> SearchMetrics:
        vector_store = getattr(memory, "vector_store", None)
        metrics = getattr(vector_store, "last_search_metrics", None)
        if isinstance(metrics, SearchMetrics):
            return metrics
        return SearchMetrics(
            search_time_ms=float(getattr(metrics, "search_time_ms", 0.0) or 0.0),
        )

    def _search_mem0_memory(self, memory: Any, query: str) -> Any:
        vector_store = getattr(memory, "vector_store", None)
        search = getattr(vector_store, "search", None)
        if not callable(search):
            raise RuntimeError("Mem0 memory has no searchable vector_store.")

        query_embedding = embed_mem0_query(memory, query)
        return search(query=query, vectors=query_embedding, top_k=self.config.top_k)


def _run_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
    logger.info("Loading LoCoMo dataset from {}", config.dataset_path)
    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    planned_questions = _planned_question_count(samples, config.max_questions)
    logger.info(
        "Loaded {} samples; planned_questions={} max_samples={} max_questions={}",
        len(samples),
        planned_questions,
        config.max_samples,
        config.max_questions,
    )
    output_path = config.run_dir / "predictions.jsonl"
    all_records: list[dict[str, Any]] = []
    question_budget = config.max_questions
    completed_questions = 0
    memory_builder = SampleMemoryBuilder(config)
    question_evaluator = QuestionEvaluator(config, clients)
    with JsonlWriter(output_path) as writer:
        for sample_index, sample in enumerate(samples, start=1):
            if question_budget is not None and question_budget <= 0:
                break
            sample_question_count = len(sample.qa)
            if question_budget is not None:
                sample_question_count = min(sample_question_count, question_budget)
            logger.info(
                "Sample {}/{} sample_id={} turns={} questions={} starting",
                sample_index,
                len(samples),
                sample.sample_id,
                len(sample.turns),
                sample_question_count,
            )
            memory = memory_builder.build(sample)
            try:
                sample_questions = sample.qa
                if question_budget is not None:
                    sample_questions = sample_questions[:question_budget]
                for qa in sample_questions:
                    next_question = completed_questions + 1
                    if _should_log_progress(next_question, planned_questions, config.log_every):
                        logger.info(
                            "Question {}/{} starting sample_id={} question_id={} category={}",
                            next_question,
                            planned_questions,
                            sample.sample_id,
                            qa.question_id,
                            qa.category,
                        )
                    record = question_evaluator.answer(sample, qa, memory)
                    writer.write(record)
                    all_records.append(record)
                    completed_questions += 1
                    if _should_log_progress(completed_questions, planned_questions, config.log_every):
                        logger.info(
                            "Question {}/{} finished sample_id={} question_id={} judge={}",
                            completed_questions,
                            planned_questions,
                            sample.sample_id,
                            qa.question_id,
                            _judge_label(record.get("judge", {}).get("correct")),
                        )
                    if question_budget is not None:
                        question_budget -= 1
                        if question_budget <= 0:
                            break
            finally:
                memory_builder.log_embedding_cache_stats(memory, sample.sample_id)
                memory_builder.close(memory)
            logger.info("Sample {}/{} sample_id={} finished", sample_index, len(samples), sample.sample_id)
    logger.info("Wrote {} prediction records to {}", len(all_records), output_path)
    return all_records


def _planned_question_count(samples: list[ConversationSample], max_questions: int | None) -> int:
    total = sum(len(sample.qa) for sample in samples)
    if max_questions is None:
        return total
    return min(total, max_questions)


def _should_log_progress(index: int, total: int, interval: int) -> bool:
    if interval <= 0 or total <= 0:
        return False
    return index == 1 or index == total or index % interval == 0


def _judge_label(value: Any) -> str:
    if value is True:
        return "correct"
    if value is False:
        return "incorrect"
    return "skipped"


def _format_accuracy(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
