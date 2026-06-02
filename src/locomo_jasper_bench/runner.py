from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .clients import (
    ChatClient,
    EmbeddingClient,
    HashEmbeddingClient,
    OpenAICompatibleChatClient,
    OpenAIEmbeddingClient,
)
from .config import BenchmarkConfig
from .data import ConversationSample, QuestionAnswer, format_turn_for_memory, load_locomo
from .jasper_store import BuildMetrics, JasperVectorStore, VectorStoreConfig
from .mem0_jasper import JasperMemory
from .prompts import build_answer_messages, build_judge_messages, parse_judge_response
from .results import JsonlWriter, read_jsonl, summarize_records, write_json
from .system import collect_system_metadata


@dataclass(slots=True)
class RuntimeClients:
    answer_client: ChatClient
    judge_client: ChatClient
    embedding_client: EmbeddingClient


def run_benchmark(config: BenchmarkConfig, clients: RuntimeClients | None = None) -> dict[str, Any]:
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
    return summary


def build_clients(config: BenchmarkConfig) -> RuntimeClients:
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
    if config.embedding_provider == "hash":
        embedding_client: EmbeddingClient = HashEmbeddingClient(config.hash_embedding_dim)
    else:
        embedding_client = OpenAIEmbeddingClient(
            api_key=config.embedding_api_key,
            model=config.embedding_model,
            base_url=config.embedding_base_url,
            batch_size=config.embedding_batch_size,
        )
    return RuntimeClients(answer_client=answer_client, judge_client=judge_client, embedding_client=embedding_client)


def _run_prediction_mode(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
    samples = load_locomo(config.dataset_path, max_samples=config.max_samples)
    output_path = config.run_dir / "predictions.jsonl"
    all_records: list[dict[str, Any]] = []
    question_budget = config.max_questions
    with JsonlWriter(output_path) as writer:
        for sample in samples:
            if question_budget is not None and question_budget <= 0:
                break
            memory, build_metrics = _build_memory_for_sample(config, clients.embedding_client, sample)
            sample_questions = sample.qa
            if question_budget is not None:
                sample_questions = sample_questions[:question_budget]
            for qa in sample_questions:
                record = _answer_question(config, clients, sample, qa, memory, build_metrics)
                writer.write(record)
                all_records.append(record)
                if question_budget is not None:
                    question_budget -= 1
                    if question_budget <= 0:
                        break
            memory.store.close()
    return all_records


def _build_memory_for_sample(
    config: BenchmarkConfig,
    embedder: EmbeddingClient,
    sample: ConversationSample,
) -> tuple[JasperMemory, BuildMetrics]:
    store_root = config.run_dir / "indexes" / sample.sample_id
    store = JasperVectorStore(store_root, _store_config(config))
    memory = JasperMemory(embedder=embedder, store=store)
    texts = [format_turn_for_memory(turn) for turn in sample.turns]
    payloads = [
        {
            "memory": text,
            "sample_id": sample.sample_id,
            "turn_id": turn.id,
            "session_id": turn.session_id,
            "speaker": turn.speaker,
            "timestamp": turn.timestamp,
            "metadata": {
                "sample_id": sample.sample_id,
                "session_id": turn.session_id,
                "turn_index": turn.turn_index,
                "speaker": turn.speaker,
            },
        }
        for turn, text in zip(sample.turns, texts)
    ]
    ids = [turn.id for turn in sample.turns]
    if texts:
        memory.add_texts(texts, payloads, ids)
    build_metrics = store.finalize()
    return memory, build_metrics


def _answer_question(
    config: BenchmarkConfig,
    clients: RuntimeClients,
    sample: ConversationSample,
    qa: QuestionAnswer,
    memory: JasperMemory,
    build_metrics: BuildMetrics,
) -> dict[str, Any]:
    started = time.perf_counter()
    memory_search = memory.search_with_metrics(qa.question, top_k=config.top_k)
    answer_messages = build_answer_messages(sample, qa, memory_search.hits)
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
            for hit in memory_search.hits
        ],
        "judge": judge_payload,
        "latency_ms": {
            "memory_search_ms": memory_search.total_time_ms,
            "memory_embedding_ms": memory_search.embedding_time_ms,
            "vector_search_ms": memory_search.store_metrics.search_time_ms,
            "answer_generation_ms": answer.latency_ms,
            "judge_ms": judge_latency_ms,
            "end_to_end_ms": end_to_end_ms,
        },
        "vllm": {
            "answer": answer.metrics(),
            "judge": judge_metrics,
        },
        "memory": {
            "backend": memory_search.store_metrics.backend,
            "embedding_time_ms": memory_search.embedding_time_ms,
            "vector_search_ms": memory_search.store_metrics.search_time_ms,
            "indexed_vector_count": memory_search.store_metrics.indexed_vector_count,
            "embedding_dim": memory_search.store_metrics.embedding_dim,
        },
        "index": {
            "backend": build_metrics.backend,
            "graph_build_time_ms": build_metrics.graph_build_time_ms,
            "indexed_vector_count": build_metrics.indexed_vector_count,
            "embedding_dim": build_metrics.embedding_dim,
            "graph_path": build_metrics.graph_path,
        },
    }


def _run_evaluate_only(config: BenchmarkConfig, clients: RuntimeClients) -> list[dict[str, Any]]:
    assert config.predictions_path is not None
    predictions_path = _resolve_predictions_path(config.predictions_path)
    predictions = read_jsonl(predictions_path)
    output_path = config.run_dir / "evaluations.jsonl"
    rows: list[dict[str, Any]] = []
    with JsonlWriter(output_path) as writer:
        for row in predictions:
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
    return rows


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


def _resolve_predictions_path(path: Path) -> Path:
    if path.is_dir():
        return path / "predictions.jsonl"
    return path
