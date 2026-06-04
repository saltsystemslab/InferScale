from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from .clients import (
    ChatClient,
    EmbeddingClient,
    OpenAICompatibleChatClient,
)
from .config import BenchmarkConfig
from .data import ConversationSample, QuestionAnswer, format_turn_for_memory, load_locomo
from .embedding_cache import CachedEmbedder
from .jasper_store import BuildMetrics, SearchMetrics, VectorStoreConfig
from .mem0_jasper import create_mem0_memory, mem0_results_to_search_hits
from .prompts import build_full_context_answer_messages, build_judge_messages, build_retrieval_answer_messages, parse_judge_response
from .results import JsonlWriter, read_jsonl, summarize_records, write_json
from .system import collect_system_metadata


@dataclass(slots=True)
class RuntimeClients:
    answer_client: ChatClient
    judge_client: ChatClient
    embedding_client: EmbeddingClient | None = None


def run_benchmark(config: BenchmarkConfig, clients: RuntimeClients | None = None) -> dict[str, Any]:
    logger.info(
        "Starting benchmark run_id={} mode={} dataset={} results_dir={}",
        config.run_id,
        config.mode,
        config.dataset_path,
        config.run_dir,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.json", config.to_jsonable())

    runtime_clients = clients or build_clients(config)
    system_metadata = collect_system_metadata(vllm_command=config.vllm_command)
    write_json(config.run_dir / "system.json", system_metadata)

    if config.mode == "evaluate-only":
        records = _run_evaluate_only(config, runtime_clients)
    else:
        records = _run_prediction_mode(config, runtime_clients)

    summary = summarize_records(
        records,
        run_id=config.run_id,
        mode=config.mode,
        config=config.to_jsonable(),
        system_metadata=system_metadata,
    )
    write_json(config.run_dir / "summary.json", summary)
    logger.info(
        "Finished benchmark run_id={} questions={} judged={} accuracy={}",
        config.run_id,
        summary["question_count"],
        summary["judged_count"],
        _format_accuracy(summary.get("accuracy")),
    )
    logger.info("Wrote results to {}", config.run_dir)
    return summary


def build_clients(config: BenchmarkConfig) -> RuntimeClients:
    logger.info(
        "Configuring clients llm={} judge={} context_mode={} embedding_provider={} vector_backend={}",
        config.llm_base_url,
        config.judge_base_url,
        config.context_mode,
        config.embedding_provider,
        config.vector_backend,
    )
    answer_client = OpenAICompatibleChatClient(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.model,
        stream=config.stream,
        extra_body=config.llm_extra_body,
    )
    judge_client = OpenAICompatibleChatClient(
        base_url=config.judge_base_url,
        api_key=config.judge_api_key,
        model=config.judge_model,
        stream=config.stream,
        extra_body=config.judge_extra_body,
    )
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client)


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
    with JsonlWriter(output_path) as writer:
        for sample_index, sample in enumerate(samples, start=1):
            if question_budget is not None and question_budget <= 0:
                break
            context_mode = _resolved_context_mode(config.context_mode)
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
            memory: Any | None
            if context_mode == "mem0":
                memory, build_metrics, index_metadata = _build_memory_for_sample(config, sample)
            else:
                logger.info(
                    "Using full conversation context for sample_id={} turns={}",
                    sample.sample_id,
                    len(sample.turns),
                )
                memory = None
                build_metrics = _empty_build_metrics()
                index_metadata = {}
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
                record = _answer_question(config, clients, sample, qa, memory, build_metrics, index_metadata)
                writer.write(record)
                all_records.append(record)
                completed_questions += 1
                if _should_log_progress(completed_questions, planned_questions, config.log_every):
                    logger.info(
                        "Question {}/{} finished sample_id={} question_id={} judge={} end_to_end_ms={:.1f}",
                        completed_questions,
                        planned_questions,
                        sample.sample_id,
                        qa.question_id,
                        _judge_label(record.get("judge", {}).get("correct")),
                        record["latency_ms"]["end_to_end_ms"],
                    )
                if question_budget is not None:
                    question_budget -= 1
                    if question_budget <= 0:
                        break
            if memory is not None:
                _log_embedding_cache_stats(memory, sample.sample_id)
                _close_mem0_memory(memory)
            logger.info("Sample {}/{} sample_id={} finished", sample_index, len(samples), sample.sample_id)
    logger.info("Wrote {} prediction records to {}", len(all_records), output_path)
    return all_records


def _build_memory_for_sample(
    config: BenchmarkConfig,
    sample: ConversationSample,
) -> tuple[Any, BuildMetrics, dict[str, Any]]:
    if config.embedding_provider != "openai":
        raise RuntimeError("Mem0 context mode currently uses mem0ai's OpenAI embedder; set --embedding-provider openai.")

    store_root = config.run_dir / "mem0" / sample.sample_id
    memory = create_mem0_memory(
        store_root=store_root,
        vector_config=_store_config(config),
        embedding_model=config.embedding_model,
        embedding_api_key=config.embedding_api_key,
        embedding_base_url=config.embedding_base_url,
    )
    _install_embedding_cache(memory, config)

    add_started = time.perf_counter()
    for turn in sample.turns:
        text = format_turn_for_memory(turn)
        metadata = {
            "user_id": sample.sample_id,
            "sample_id": sample.sample_id,
            "turn_id": turn.id,
            "session_id": turn.session_id,
            "turn_index": turn.turn_index,
            "speaker": turn.speaker,
            "timestamp": turn.timestamp,
        }
        memory.add(
            [{"role": "user", "content": text}],
            user_id=sample.sample_id,
            infer=False,
            metadata=metadata,
        )
    add_time_ms = (time.perf_counter() - add_started) * 1000
    logger.info(
        "Added {} LoCoMo turns to Mem0 for sample_id={} infer=false add_ms={:.1f}",
        len(sample.turns),
        sample.sample_id,
        add_time_ms,
    )

    logger.info("Building {} index for sample_id={}", config.vector_backend, sample.sample_id)
    build_metrics = _finalize_mem0_memory(memory)
    logger.info(
        "Index ready sample_id={} backend={} vectors={} dim={} build_ms={:.1f}",
        sample.sample_id,
        build_metrics.backend,
        build_metrics.indexed_vector_count,
        build_metrics.embedding_dim,
        build_metrics.graph_build_time_ms,
    )
    return memory, build_metrics, {
        "mem0_path": str(store_root),
        "memory_add_time_ms": add_time_ms,
        "memory_add_count": len(sample.turns),
        "infer": False,
    }


def _answer_question(
    config: BenchmarkConfig,
    clients: RuntimeClients,
    sample: ConversationSample,
    qa: QuestionAnswer,
    memory: Any | None,
    build_metrics: BuildMetrics,
    index_metadata: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    if _resolved_context_mode(config.context_mode) == "mem0":
        if memory is None:
            raise RuntimeError("Mem0 context mode requires a memory store.")
        memory_search_started = time.perf_counter()
        raw_results = _search_mem0_memory(memory, qa.question, sample.sample_id, config.top_k)
        memory_search_ms = (time.perf_counter() - memory_search_started) * 1000
        hits = mem0_results_to_search_hits(raw_results)
        answer_messages = build_retrieval_answer_messages(sample, qa, hits)
        memory_embedding_ms = None
        store_metrics = _mem0_store_search_metrics(memory)
        vector_search_ms = store_metrics.search_time_ms
        memory_backend = f"mem0-{store_metrics.backend}"
        indexed_vector_count = store_metrics.indexed_vector_count
        embedding_dim = store_metrics.embedding_dim
    else:
        hits = []
        answer_messages = build_full_context_answer_messages(sample, qa)
        memory_search_ms = 0.0
        memory_embedding_ms = None
        vector_search_ms = 0.0
        memory_backend = "none"
        indexed_vector_count = 0
        embedding_dim = None
    answer = clients.answer_client.chat(
        answer_messages,
        max_tokens=config.max_answer_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    judge_payload: dict[str, Any] = {"correct": None, "reason": None, "raw": None}
    judge_metrics: dict[str, Any] | None = None
    judge_latency_ms = None
    if not config.skip_judge:
        judge_messages = build_judge_messages(qa, answer.content)
        judge = clients.judge_client.chat(
            judge_messages,
            max_tokens=config.max_judge_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        correct, reason = parse_judge_response(judge.content)
        judge_payload = {"correct": correct, "reason": reason, "raw": judge.content}
        judge_metrics = judge.metrics()
        judge_latency_ms = judge.latency_ms
    end_to_end_ms = (time.perf_counter() - started) * 1000

    return {
        "run_id": config.run_id,
        "mode": config.mode,
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
        "latency_ms": {
            "memory_search_ms": memory_search_ms,
            "memory_embedding_ms": memory_embedding_ms,
            "vector_search_ms": vector_search_ms,
            "answer_generation_ms": answer.latency_ms,
            "judge_ms": judge_latency_ms,
            "end_to_end_ms": end_to_end_ms,
        },
        "vllm": {
            "answer": answer.metrics(),
            "judge": judge_metrics,
        },
        "memory": {
            "backend": memory_backend,
            "embedding_time_ms": memory_embedding_ms,
            "vector_search_ms": vector_search_ms,
            "indexed_vector_count": indexed_vector_count,
            "embedding_dim": embedding_dim,
        },
        "index": {
            "backend": build_metrics.backend,
            "graph_build_time_ms": build_metrics.graph_build_time_ms,
            "indexed_vector_count": build_metrics.indexed_vector_count,
            "embedding_dim": build_metrics.embedding_dim,
            "graph_path": build_metrics.graph_path,
            **index_metadata,
        },
    }


def _run_evaluate_only(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
    assert config.predictions_path is not None
    predictions_path = _resolve_predictions_path(config.predictions_path)
    logger.info("Loading saved predictions from {}", predictions_path)
    predictions = read_jsonl(predictions_path)
    logger.info("Loaded {} predictions for re-judging", len(predictions))
    output_path = config.run_dir / "evaluations.jsonl"
    rows: list[dict[str, Any]] = []
    with JsonlWriter(output_path) as writer:
        for index, row in enumerate(predictions, start=1):
            if _should_log_progress(index, len(predictions), config.log_every):
                logger.info(
                    "Evaluation {}/{} starting sample_id={} question_id={}",
                    index,
                    len(predictions),
                    row.get("sample_id"),
                    row.get("question_id"),
                )
            qa = QuestionAnswer(
                sample_id=str(row.get("sample_id") or ""),
                question_id=str(row.get("question_id") or ""),
                question=str(row.get("question") or ""),
                answer=str(row.get("gold_answer") or row.get("answer") or ""),
                category=str(row.get("category") or "unknown"),
                evidence=row.get("evidence"),
            )
            judge = clients.judge_client.chat(
                build_judge_messages(qa, str(row.get("predicted_answer") or "")),
                max_tokens=config.max_judge_tokens,
                temperature=0.0,
                top_p=1.0,
            )
            correct, reason = parse_judge_response(judge.content)
            updated = dict(row)
            updated["run_id"] = config.run_id
            updated["mode"] = "evaluate-only"
            updated["judge"] = {"correct": correct, "reason": reason, "raw": judge.content}
            updated.setdefault("vllm", {})
            updated["vllm"]["judge"] = judge.metrics()
            updated.setdefault("latency_ms", {})
            updated["latency_ms"]["judge_ms"] = judge.latency_ms
            writer.write(updated)
            rows.append(updated)
            if _should_log_progress(index, len(predictions), config.log_every):
                logger.info(
                    "Evaluation {}/{} finished sample_id={} question_id={} judge={} judge_ms={:.1f}",
                    index,
                    len(predictions),
                    updated.get("sample_id"),
                    updated.get("question_id"),
                    _judge_label(correct),
                    judge.latency_ms,
                )
    logger.info("Wrote {} evaluation records to {}", len(rows), output_path)
    return rows


def _resolved_context_mode(context_mode: str) -> str:
    if context_mode == "retrieval":
        return "mem0"
    return context_mode


def _finalize_mem0_memory(memory: Any) -> BuildMetrics:
    vector_store = getattr(memory, "vector_store", None)
    finalize = getattr(vector_store, "finalize", None)
    if callable(finalize):
        metrics = finalize()
        if isinstance(metrics, BuildMetrics):
            return metrics
        if isinstance(metrics, dict):
            return BuildMetrics(
                backend=str(metrics.get("backend") or "jasper"),
                graph_build_time_ms=float(metrics.get("graph_build_time_ms") or 0.0),
                indexed_vector_count=int(metrics.get("indexed_vector_count") or 0),
                embedding_dim=metrics.get("embedding_dim"),
                graph_path=metrics.get("graph_path"),
            )
    return BuildMetrics(
        backend="jasper",
        graph_build_time_ms=0.0,
        indexed_vector_count=0,
        embedding_dim=None,
        graph_path=None,
    )


def _mem0_store_search_metrics(memory: Any) -> SearchMetrics:
    vector_store = getattr(memory, "vector_store", None)
    metrics = getattr(vector_store, "last_search_metrics", None)
    if isinstance(metrics, SearchMetrics):
        return metrics
    store = getattr(vector_store, "store", None)
    return SearchMetrics(
        backend=getattr(getattr(store, "config", None), "backend", "jasper"),
        search_time_ms=float(getattr(metrics, "search_time_ms", 0.0) or 0.0),
        indexed_vector_count=int(getattr(store, "vector_count", 0) or 0),
        embedding_dim=getattr(store, "dim", None),
    )


def _search_mem0_memory(memory: Any, query: str, sample_id: str, top_k: int) -> Any:
    direct_results = _search_mem0_vector_store_direct(memory, query, top_k)
    if direct_results is not None:
        return direct_results
    try:
        return memory.search(query=query, filters={"user_id": sample_id}, top_k=top_k)
    except TypeError as exc:
        if "top_k" not in str(exc):
            raise
    return memory.search(query=query, filters={"user_id": sample_id}, limit=top_k)


def _search_mem0_vector_store_direct(memory: Any, query: str, top_k: int) -> Any:
    vector_store = getattr(memory, "vector_store", None)
    search = getattr(vector_store, "search", None)
    if not callable(search):
        return None

    query_embedding = _embed_mem0_query(memory, query)
    if query_embedding is None:
        return None

    try:
        return search(query=query, vectors=query_embedding, top_k=top_k)
    except TypeError as exc:
        if "top_k" not in str(exc):
            raise
    return search(query=query, vectors=query_embedding, limit=top_k)


def _embed_mem0_query(memory: Any, query: str) -> Any:
    embedder = getattr(memory, "embedding_model", None) or getattr(memory, "embedder", None)
    embed = getattr(embedder, "embed", None)
    if not callable(embed):
        return None
    try:
        return embed(query, "search")
    except TypeError:
        return embed(query)


def _install_embedding_cache(memory: Any, config: BenchmarkConfig) -> None:
    if not config.embedding_cache_enabled:
        logger.info("Embedding cache disabled")
        return

    embedder = getattr(memory, "embedding_model", None) or getattr(memory, "embedder", None)
    if embedder is None:
        logger.warning("Embedding cache requested but Mem0 memory has no embedder attribute")
        return

    cached = CachedEmbedder(embedder, cache_dir=config.embedding_cache_dir, model=config.embedding_model)
    if hasattr(memory, "embedding_model"):
        memory.embedding_model = cached
    if hasattr(memory, "embedder"):
        memory.embedder = cached
    memory._locomo_embedding_cache = cached
    logger.info("Embedding cache enabled dir={}", cached.cache_dir)


def _log_embedding_cache_stats(memory: Any, sample_id: str) -> None:
    cache = getattr(memory, "_locomo_embedding_cache", None)
    if not isinstance(cache, CachedEmbedder):
        return
    stats = cache.stats()
    logger.info(
        "Embedding cache sample_id={} hits={} misses={} dir={}",
        sample_id,
        stats["hits"],
        stats["misses"],
        stats["cache_dir"],
    )


def _close_mem0_memory(memory: Any) -> None:
    vector_store = getattr(memory, "vector_store", None)
    close = getattr(vector_store, "close", None)
    if callable(close):
        close()


def _store_config(config: BenchmarkConfig) -> VectorStoreConfig:
    return VectorStoreConfig(
        backend=config.vector_backend,
        distance=config.vector_distance,
        normalize=config.normalize_embeddings,
        n_neighbors=config.jasper_n_neighbors,
        alpha=config.jasper_alpha,
        workspace_budget=config.jasper_workspace_budget,
        beam_width=config.jasper_beam_width,
    )


def _empty_build_metrics() -> BuildMetrics:
    return BuildMetrics(
        backend="none",
        graph_build_time_ms=0.0,
        indexed_vector_count=0,
        embedding_dim=None,
        graph_path=None,
    )


def _resolve_predictions_path(path: Path) -> Path:
    if path.is_dir():
        return path / "predictions.jsonl"
    return path


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
