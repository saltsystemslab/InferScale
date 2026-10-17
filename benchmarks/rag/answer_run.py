"""The run stage: retrieve, answer, and judge every query, then write the run's files."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from benchmarks.common.answers import final_answer_text
from benchmarks.common.clients import ChatClient
from benchmarks.common.files import JsonlWriter, write_json
from benchmarks.common.judge import format_accuracy, skipped_judge_payload
from benchmarks.common.system import collect_system_metadata
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.corpus import RagCorpus, load_rag_corpus
from benchmarks.rag.data_types import RagQuery
from benchmarks.rag.embedder import build_cached_embedder
from benchmarks.rag.engine_client import RagAnswerClient
from benchmarks.rag.evaluation import build_query_record
from benchmarks.rag.judging import judge_client_for, judge_metadata, judge_rag_answer
from benchmarks.rag.metrics import retrieval_metrics_for_query
from benchmarks.rag.retrieval import CorpusRetriever
from benchmarks.rag.run_outputs import (
    CONFIG_FILE,
    PREDICTIONS_FILE,
    SETUP_FILE,
    SYSTEM_FILE,
    write_run_outputs,
)


def run_answer(config: RagBenchConfig) -> dict[str, Any]:
    started = time.perf_counter()
    corpus = load_rag_corpus(config)
    queries = corpus.queries[: config.max_queries] if config.max_queries else corpus.queries
    logger.info(
        "Starting RAG run_id={} dataset={} mode={} model={} docs={} chunks={} queries={}",
        config.run_id,
        config.dataset_name,
        config.result_mode(),
        config.model,
        len(corpus.docs),
        len(corpus.chunks),
        len(queries),
    )
    # The client validates its caches before anything is written for this run.
    client = build_answer_client(config, corpus)
    setup_metrics: dict[str, Any] = {
        "dataset": config.dataset_name,
        "doc_count": len(corpus.docs),
        "chunk_count": len(corpus.chunks),
        "corpus_total_tokens": corpus.total_tokens,
    }
    retriever: CorpusRetriever | None = None
    records: list[dict[str, Any]] = []
    try:
        judge_client = _judge_client(config)
        config.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(config.run_dir / CONFIG_FILE, config.to_jsonable())
        system_metadata = collect_system_metadata()
        write_json(config.run_dir / SYSTEM_FILE, system_metadata)
        retriever = CorpusRetriever(
            config=config,
            chunks=corpus.chunks,
            embedder=build_cached_embedder(config, mode="read"),
        )
        setup_metrics.update(retriever.build())
        client.start_llm()
        records = _answer_queries(config, queries, corpus, retriever, client, judge_client)
    finally:
        setup_metrics.update(client.setup_stats())
        client.close()
        if retriever is not None:
            retriever.close()
    setup_metrics["run_wall_time_ms"] = (time.perf_counter() - started) * 1000
    summary = write_run_outputs(
        config,
        records,
        saved_config=config.to_jsonable(),
        system_metadata=system_metadata,
        setup_metrics=setup_metrics,
        write_reports=True,
    )
    write_json(config.run_dir / SETUP_FILE, setup_metrics)
    logger.info(
        "Finished RAG run_id={} questions={} judged={} accuracy={} em={} f1={}",
        config.run_id,
        summary["question_count"],
        summary["judged_count"],
        format_accuracy(summary["metrics"].get("accuracy")),
        format_accuracy(summary["metrics"].get("exact_match")),
        format_accuracy(summary["metrics"].get("f1")),
    )
    return summary


def build_answer_client(config: RagBenchConfig, corpus: RagCorpus) -> RagAnswerClient:
    """The client for the configured answer backend; the KV client needs a complete chunk cache."""
    if config.answer_backend == "kv-injection":
        from benchmarks.rag.answer_kv import RagKvAnswerClient
        from benchmarks.rag.kv_cache import RagChunkCache

        cache = RagChunkCache.for_config(config, corpus.docs)
        cache.require_complete(corpus.chunks)
        return RagKvAnswerClient(
            config, chunks=corpus.chunks, cache=cache, prompt_profile=corpus.spec.prompt_profile
        )
    from benchmarks.rag.answer_prompt_injection import RagPromptInjectionAnswerClient

    return RagPromptInjectionAnswerClient(
        config, chunks_by_id=corpus.chunks_by_id, prompt_profile=corpus.spec.prompt_profile
    )


def _judge_client(config: RagBenchConfig) -> ChatClient | None:
    judge_client = judge_client_for(config)
    if not config.skip_judge and judge_client is None:
        raise RuntimeError(
            "Judge client is not configured. Set skip_judge=true in the run JSON to write "
            "unjudged predictions."
        )
    return judge_client


def _answer_queries(
    config: RagBenchConfig,
    queries: list[RagQuery],
    corpus: RagCorpus,
    retriever: CorpusRetriever,
    client: RagAnswerClient,
    judge_client: ChatClient | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with JsonlWriter(config.run_dir / PREDICTIONS_FILE) as writer:
        for index, query in enumerate(queries, start=1):
            record = _answer_query(config, query, corpus, retriever, client, judge_client)
            records.append(record)
            writer.write(record)
            if index % max(1, config.log_every) == 0 or index == len(queries):
                logger.info(
                    "Answered {}/{} queries (last: {} {})",
                    index,
                    len(queries),
                    query.query_id,
                    query.question_type,
                )
    return records


def _answer_query(
    config: RagBenchConfig,
    query: RagQuery,
    corpus: RagCorpus,
    retriever: CorpusRetriever,
    client: RagAnswerClient,
    judge_client: ChatClient | None,
) -> dict[str, Any]:
    query_started = time.perf_counter()
    hits, retrieval_metrics = retriever.search(query.question, top_k=config.top_k)
    answer = client.answer(query, hits, query_started_at=query_started)
    answer.content = final_answer_text(answer.content)
    if judge_client is None:
        judge_payload = skipped_judge_payload(judge_metadata(config))
    else:
        judge_payload = judge_rag_answer(
            config,
            judge_client,
            question=query.question,
            gold_answers=query.gold_answers,
            predicted_answer=answer.content,
        )
    return build_query_record(
        config,
        query,
        hits,
        answer,
        retrieval_metrics=retrieval_metrics,
        retrieval_quality=retrieval_metrics_for_query(query, hits, corpus.chunk_text_by_id),
        judge_payload=judge_payload,
        docs_by_id=corpus.docs_by_id,
    )
