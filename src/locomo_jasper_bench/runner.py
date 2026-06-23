from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from .clients import ChatClient, OpenAICompatibleChatClient
from .config import BenchmarkConfig
from .data import ConversationSample, QuestionAnswer, load_locomo
from .prompts import build_judge_messages, build_retrieval_answer_messages, parse_judge_response
from .results import JsonlWriter, summarize_records, write_json
from .system import collect_system_metadata
from .vector_types import SearchHit, SearchMetrics


@dataclass(slots=True)
class RuntimeClients:
    answer_client: ChatClient
    judge_client: ChatClient | None


@dataclass(slots=True)
class PreparedQuestion:
    qa: QuestionAnswer
    hits: list[SearchHit]
    store_metrics: SearchMetrics


@dataclass(slots=True)
class PreparedSample:
    index: int
    sample: ConversationSample
    questions: list[PreparedQuestion]


def run_benchmark(config: BenchmarkConfig, clients: RuntimeClients | None = None) -> dict[str, Any]:
    logger.info(
        "Starting benchmark run_id={} dataset={} results_dir={}",
        config.run_id,
        config.dataset_path,
        config.run_dir,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.json", config.to_jsonable())

    owns_clients = clients is None
    runtime_clients = clients or build_clients(config)
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / "system.json", system_metadata)

    try:
        records = _run_prediction_mode(config, runtime_clients)
    finally:
        if owns_clients:
            close_answer_client = getattr(runtime_clients.answer_client, "close", None)
            if callable(close_answer_client):
                close_answer_client()

    summary = summarize_records(
        records,
        run_id=config.run_id,
        mode=_result_mode(config),
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
    if config.answer_backend == "vllm-kv":
        from .kv.answer_client import VLLMChunkedKVAnswerClient

        answer_client = VLLMChunkedKVAnswerClient(config)
    elif config.answer_backend == "vllm-prefix":
        from .kv.prefix_answer_client import VLLMPrefixPromptAnswerClient

        answer_client = VLLMPrefixPromptAnswerClient(config)
    else:
        answer_client = OpenAICompatibleChatClient(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
            model=config.model,
            stream=config.stream,
        )
    if config.skip_judge:
        judge_client = None
    else:
        judge_client = OpenAICompatibleChatClient(
            base_url=config.judge_base_url,
            api_key=config.judge_api_key,
            model=config.judge_model,
            stream=False,
        )
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client)


def judge_existing_run(config: BenchmarkConfig) -> dict[str, Any]:
    predictions_path = config.run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    logger.info("Judging existing run_id={} predictions={}", config.run_id, predictions_path)
    records = _read_jsonl(predictions_path)
    saved_config = _read_json_or_default(config.run_dir / "config.json", config.to_jsonable())
    system_metadata = _read_json_or_default(config.run_dir / "system.json", {})
    judge_client = OpenAICompatibleChatClient(
        base_url=config.judge_base_url,
        api_key=config.judge_api_key,
        model=config.judge_model,
        stream=False,
    )

    judged_now = 0
    summary: dict[str, Any] | None = None
    for row_number, record in enumerate(records, start=1):
        if _is_judged(record):
            continue
        try:
            judge_payload = _judge_record(config, judge_client, record)
        except Exception as exc:
            record["judge"] = _failed_judge_payload(exc)
            _write_deferred_judging_outputs(
                config,
                predictions_path,
                records,
                saved_config=saved_config,
                system_metadata=system_metadata,
            )
            raise RuntimeError(
                f"Judge request failed for row {row_number}/{len(records)} {_record_label(record)}. "
                f"Saved progress to {predictions_path}; fix or restart the judge server, then rerun --judge-only. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        record["judge"] = judge_payload
        judged_now += 1
        summary = _write_deferred_judging_outputs(
            config,
            predictions_path,
            records,
            saved_config=saved_config,
            system_metadata=system_metadata,
        )

    if summary is None:
        summary = _write_deferred_judging_outputs(
            config,
            predictions_path,
            records,
            saved_config=saved_config,
            system_metadata=system_metadata,
        )
    logger.info(
        "Finished deferred judging run_id={} judged_now={} judged={} accuracy={}",
        config.run_id,
        judged_now,
        summary["judged_count"],
        _format_accuracy(summary.get("metrics", {}).get("accuracy")),
    )
    return summary


class QuestionEvaluator:
    def __init__(self, config: BenchmarkConfig, clients: RuntimeClients) -> None:
        self.config = config
        self.clients = clients

    def answer(self, sample: ConversationSample, qa: QuestionAnswer, memory: Any | None) -> dict[str, Any]:
        retrieval_started_at = time.perf_counter()
        if memory is None:
            raise RuntimeError("Mem0 context requires a memory store.")
        hits = self._search_mem0_memory(memory, qa.question)
        store_metrics = self._mem0_store_search_metrics(memory)
        return self.answer_from_hits(
            sample,
            qa,
            hits,
            store_metrics,
            retrieval_started_at=retrieval_started_at,
        )

    def answer_from_hits(
        self,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        store_metrics: SearchMetrics,
        *,
        ttft_started_at: float | None = None,
        retrieval_started_at: float | None = None,
    ) -> dict[str, Any]:
        kv_answer = getattr(self.clients.answer_client, "answer_with_retrieved_memory", None)
        if callable(kv_answer):
            answer = kv_answer(
                sample=sample,
                qa=qa,
                hits=hits,
                max_tokens=self.config.max_answer_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                ttft_started_at=ttft_started_at,
            )
        else:
            answer_messages = build_retrieval_answer_messages(sample, qa, hits)
            llm_started_at = time.perf_counter()
            answer = self.clients.answer_client.chat(
                answer_messages,
                max_tokens=self.config.max_answer_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                ttft_started_at=llm_started_at,
            )
            if retrieval_started_at is not None and answer.ttft_ms is not None:
                answer.metrics["retrieval_to_ttft_ms"] = (
                    (llm_started_at - retrieval_started_at) * 1000 + answer.ttft_ms
                )
        return self.record_answer(sample, qa, hits, store_metrics, answer)

    def record_answer(
        self,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        store_metrics: SearchMetrics,
        answer: Any,
    ) -> dict[str, Any]:
        if self.config.skip_judge:
            judge_payload = _skipped_judge_payload()
        else:
            if self.clients.judge_client is None:
                raise RuntimeError("Judge client is not configured. Use --skip-judge to write unjudged predictions.")
            judge_payload = _judge_qa(self.config, self.clients.judge_client, qa, answer.content)

        return {
            "run_id": self.config.run_id,
            "mode": _result_mode(self.config),
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
                "vector_db_query_time_ms": store_metrics.search_time_ms,
                **getattr(answer, "metrics", {}),
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

    def _search_mem0_memory(self, memory: Any, query: str) -> list[SearchHit]:
        from .memory_builder import embed_mem0_query

        vector_store = getattr(memory, "vector_store", None)
        search = getattr(vector_store, "search", None)
        if not callable(search):
            raise RuntimeError("Mem0 memory has no searchable vector_store.")

        query_embedding = embed_mem0_query(memory, query)
        return search(query=query, vectors=query_embedding, top_k=self.config.top_k)


def _run_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
    if config.answer_backend in {"vllm-kv", "vllm-prefix"}:
        return _run_kv_prediction_mode(config, clients)
    return _run_openai_prediction_mode(config, clients)


def _run_openai_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
    from .memory_builder import SampleMemoryBuilder

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


def _run_kv_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
    from .memory_builder import SampleMemoryBuilder

    logger.info("Loading LoCoMo dataset from {}", config.dataset_path)
    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    planned_questions = _planned_question_count(samples, config.max_questions)
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

    with JsonlWriter(output_path) as writer:
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
                    hits = question_evaluator._search_mem0_memory(memory, qa.question)
                    store_metrics = question_evaluator._mem0_store_search_metrics(memory)
                    prepared_questions.append(PreparedQuestion(qa=qa, hits=hits, store_metrics=store_metrics))
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
            prepared = PreparedSample(index=sample_index, sample=sample, questions=prepared_questions)

            if active_sample_gpu_cache:
                precompute_sample_cache(sample)
            prepared_samples.append(prepared)

        if active_sample_gpu_cache:
            if prepared_samples:
                logger.info(
                    "Precomputed GPU-resident KV caches for {} samples and {} questions; "
                    "starting one vLLM instance",
                    len(prepared_samples),
                    sum(len(prepared.questions) for prepared in prepared_samples),
                )
            else:
                logger.info("No GPU-resident KV caches were precomputed; skipping vLLM startup")
        else:
            logger.info(
                "Prepared {} samples and {} questions before vLLM startup",
                len(prepared_samples),
                sum(len(prepared.questions) for prepared in prepared_samples),
            )
        if prepared_samples:
            start_llm()
        for prepared in prepared_samples:
            sample = prepared.sample
            prepare_sample(sample, [(question.qa, question.hits) for question in prepared.questions])
            try:
                completed_questions = _answer_prepared_kv_sample(
                    config=config,
                    writer=writer,
                    question_evaluator=question_evaluator,
                    prepared=prepared,
                    total_samples=len(samples),
                    planned_questions=planned_questions,
                    completed_questions=completed_questions,
                    all_records=all_records,
                )
            finally:
                close_sample()

    logger.info("Wrote {} prepared vLLM prediction records to {}", len(all_records), output_path)
    return all_records


def _answer_prepared_kv_sample(
    *,
    config: BenchmarkConfig,
    writer: JsonlWriter,
    question_evaluator: QuestionEvaluator,
    prepared: PreparedSample,
    total_samples: int,
    planned_questions: int,
    completed_questions: int,
    all_records: list[dict[str, Any]],
) -> int:
    sample = prepared.sample
    for question in prepared.questions:
        qa = question.qa
        next_question = completed_questions + 1
        if _should_log_progress(next_question, planned_questions, config.log_every):
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
            question.store_metrics,
            ttft_started_at=time.perf_counter(),
        )
        writer.write(record)
        all_records.append(record)
        completed_questions += 1
        if _should_log_progress(completed_questions, planned_questions, config.log_every):
            logger.info(
                "KV question {}/{} finished sample_id={} question_id={} judge={}",
                completed_questions,
                planned_questions,
                sample.sample_id,
                qa.question_id,
                _judge_label(record.get("judge", {}).get("correct")),
            )
    logger.info(
        "KV sample {}/{} sample_id={} finished",
        prepared.index,
        total_samples,
        sample.sample_id,
    )
    return completed_questions


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


def _result_mode(config: BenchmarkConfig) -> str:
    if config.answer_backend == "openai":
        return "baseline"
    return config.answer_backend


def _skipped_judge_payload() -> dict[str, Any]:
    return {"correct": None, "reason": "skipped", "raw": "", "status": "skipped"}


def _failed_judge_payload(exc: Exception) -> dict[str, Any]:
    return {
        "correct": None,
        "reason": f"{type(exc).__name__}: {exc}",
        "raw": "",
        "status": "error",
    }


def _judge_qa(
    config: BenchmarkConfig,
    judge_client: ChatClient,
    qa: QuestionAnswer,
    predicted_answer: str,
) -> dict[str, Any]:
    judge_messages = build_judge_messages(qa, predicted_answer)
    judge = judge_client.chat(
        judge_messages,
        max_tokens=config.max_judge_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    correct, reason = parse_judge_response(judge.content)
    return {"correct": correct, "reason": reason, "raw": judge.content}


def _judge_record(config: BenchmarkConfig, judge_client: ChatClient, record: dict[str, Any]) -> dict[str, Any]:
    qa = QuestionAnswer(
        sample_id=str(record.get("sample_id") or ""),
        question_id=str(record.get("question_id") or ""),
        question=str(record.get("question") or ""),
        answer=str(record.get("gold_answer") or ""),
        category=str(record.get("category") or ""),
        evidence=record.get("evidence"),
    )
    return _judge_qa(config, judge_client, qa, str(record.get("predicted_answer") or ""))


def _is_judged(record: dict[str, Any]) -> bool:
    judge = record.get("judge")
    return isinstance(judge, dict) and isinstance(judge.get("correct"), bool)


def _record_label(record: dict[str, Any]) -> str:
    return (
        f"sample_id={record.get('sample_id') or ''} "
        f"question_id={record.get('question_id') or ''} "
        f"category={record.get('category') or ''}"
    ).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def _replace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with JsonlWriter(tmp_path) as writer:
        for row in rows:
            writer.write(row)
    tmp_path.replace(path)


def _write_deferred_judging_outputs(
    config: BenchmarkConfig,
    predictions_path: Path,
    records: list[dict[str, Any]],
    *,
    saved_config: dict[str, Any],
    system_metadata: dict[str, Any],
) -> dict[str, Any]:
    _replace_jsonl(predictions_path, records)
    summary = summarize_records(
        records,
        run_id=config.run_id,
        mode=_existing_run_mode(saved_config, records, config),
        config=saved_config,
        system_metadata=system_metadata,
    )
    write_json(config.run_dir / "summary.json", summary)
    return summary


def _read_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return default
    return data


def _existing_run_mode(
    saved_config: dict[str, Any],
    records: list[dict[str, Any]],
    fallback_config: BenchmarkConfig,
) -> str:
    answer_backend = saved_config.get("answer_backend")
    if answer_backend == "openai":
        return "baseline"
    if isinstance(answer_backend, str) and answer_backend:
        return answer_backend
    if records and isinstance(records[0].get("mode"), str):
        return str(records[0]["mode"])
    return _result_mode(fallback_config)
