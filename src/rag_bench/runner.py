from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from loguru import logger

from locomo_jasper_bench.reporting import write_csv
from locomo_jasper_bench.results import JsonlWriter, write_json
from locomo_jasper_bench.run_files import read_json_or_default, read_jsonl, replace_jsonl
from locomo_jasper_bench.system import collect_system_metadata

from .chunking import chunk_corpus, corpus_fingerprint
from .config import RagBenchConfig
from .data_types import RagChunk, RagDocument, RagQuery
from .datasets import get_dataset
from .evaluation import build_query_record, final_answer_text
from .judging import (
    build_judge_client,
    failed_judge_payload,
    format_accuracy,
    is_judged,
    judge_rag_answer,
    judge_rag_record,
    record_label,
    skipped_judge_payload,
)
from .kv_cache import cache_meta_base, missing_chunk_files, rag_chunk_cache_dir
from .metrics import retrieval_metrics_for_query
from .results import QUERY_METRICS_COLUMNS, build_query_metric_rows, summarize_rag_records
from .tokenizer import load_rag_tokenizer

# KV geometry (layers, kv_heads, head_dim) for the configured answer models,
# used only by --estimate-only projections. Unknown models fall back to
# transformers AutoConfig when available.
_KV_GEOMETRY = {
    "meta-llama/Llama-3.1-8B-Instruct": (32, 8, 128),
    "mistralai/Mistral-7B-Instruct-v0.3": (32, 8, 128),
    "Qwen/Qwen2.5-7B-Instruct": (28, 4, 128),
    "Qwen/Qwen3-14B": (40, 8, 128),
}
_DTYPE_BYTES = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "fp32": 4,
    "float32": 4,
}


def run_estimate(config: RagBenchConfig) -> dict[str, Any]:
    """CPU-only go/no-go gate: corpus, chunk, embedding, and KV size projections."""
    spec = get_dataset(config.dataset_name)
    docs, queries = spec.load(config.data_dir)
    tokenizer = load_rag_tokenizer(config.model, allow_transformers_fallback=True)
    chunks = chunk_corpus(docs, tokenizer=tokenizer, chunk_size=config.chunk_size)
    total_tokens = sum(chunk.token_count for chunk in chunks)
    chunk_counts = [chunk.token_count for chunk in chunks]
    doc_chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        doc_chunk_counts[chunk.doc_id] = doc_chunk_counts.get(chunk.doc_id, 0) + 1

    kv_bytes_per_token = _kv_bytes_per_token(config)
    projected_kv_bytes = kv_bytes_per_token * total_tokens if kv_bytes_per_token else None
    per_query_fetch_bytes = (
        kv_bytes_per_token * config.top_k * config.chunk_size if kv_bytes_per_token else None
    )
    encode_input_tokens = sum(
        min(
            (min(config.context_window, chunk.chunk_index) * config.chunk_size)
            + chunk.token_count,
            config.kv_max_position,
        )
        for chunk in chunks
    )

    estimate = {
        "dataset": config.dataset_name,
        "model": config.model,
        "chunk_size": config.chunk_size,
        "context_window": config.context_window,
        "top_k": config.top_k,
        "doc_count": len(docs),
        "query_count": len(queries),
        "chunk_count": len(chunks),
        "corpus_total_tokens": total_tokens,
        "chunk_tokens_min": min(chunk_counts),
        "chunk_tokens_max": max(chunk_counts),
        "chunks_per_doc_max": max(doc_chunk_counts.values()),
        "embedding_texts": len(chunks) + len(queries),
        "kv_bytes_per_token": kv_bytes_per_token,
        "projected_kv_cache_bytes": projected_kv_bytes,
        "per_query_fetch_bytes": per_query_fetch_bytes,
        "kv_encode_input_tokens": encode_input_tokens,
        "corpus_fingerprint": corpus_fingerprint(docs),
    }
    print(
        f"dataset={estimate['dataset']} model={estimate['model']} docs={estimate['doc_count']} "
        f"queries={estimate['query_count']} chunks={estimate['chunk_count']} "
        f"corpus_tokens={estimate['corpus_total_tokens']}"
    )
    print(
        f"chunk_size={config.chunk_size} context_window={config.context_window} "
        f"top_k={config.top_k} chunk_tokens_min={estimate['chunk_tokens_min']} "
        f"chunks_per_doc_max={estimate['chunks_per_doc_max']}"
    )
    if projected_kv_bytes is not None:
        print(
            f"kv_bytes_per_token={kv_bytes_per_token} "
            f"projected_kv_cache_gib={projected_kv_bytes / 1024**3:.1f} "
            f"per_query_h2d_gib={per_query_fetch_bytes / 1024**3:.2f}"
        )
        print(
            f"host_ram_required_gib={projected_kv_bytes / 1024**3:.1f} "
            "(the cpu store loads the full corpus KV into host RAM at answer time)"
        )
    else:
        print("kv_bytes_per_token=unknown (model not in the geometry table; no AutoConfig)")
    print(
        f"embedding_texts={estimate['embedding_texts']} "
        f"kv_encode_input_tokens={encode_input_tokens}"
    )
    # Machine-readable line consumed by scripts/rag/precompute_kv.sh's df check.
    print(
        "ESTIMATE "
        f"chunks={estimate['chunk_count']} "
        f"projected_kv_cache_bytes={projected_kv_bytes if projected_kv_bytes is not None else 0}"
    )
    return estimate


def run_answer(config: RagBenchConfig) -> dict[str, Any]:
    started = time.perf_counter()
    spec = get_dataset(config.dataset_name)
    docs, queries = spec.load(config.data_dir)
    if config.max_queries is not None:
        queries = queries[: config.max_queries]
    tokenizer = load_rag_tokenizer(config.model)
    chunks = chunk_corpus(docs, tokenizer=tokenizer, chunk_size=config.chunk_size)
    chunks_by_id: dict[str, RagChunk] = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
    docs_by_id: dict[str, RagDocument] = {doc.doc_id: doc for doc in docs}
    logger.info(
        "Starting RAG run_id={} dataset={} mode={} model={} docs={} chunks={} queries={}",
        config.run_id,
        config.dataset_name,
        config.result_mode(),
        config.model,
        len(docs),
        len(chunks),
        len(queries),
    )

    cache_dir: Path | None = None
    meta_base: dict[str, Any] | None = None
    if config.answer_backend == "vllm-kv":
        fingerprint = corpus_fingerprint(docs)
        cache_dir = rag_chunk_cache_dir(
            model=config.model,
            dtype=config.kv_dtype,
            chunk_size=config.chunk_size,
            context_window=config.context_window,
            max_position=config.kv_max_position,
            corpus_fingerprint=fingerprint,
            cache_root=config.kv_chunk_cache_root,
        )
        meta_base = cache_meta_base(
            dataset=config.dataset_name,
            model=config.model,
            dtype=config.kv_dtype,
            chunk_size=config.chunk_size,
            context_window=config.context_window,
            max_position=config.kv_max_position,
            corpus_fingerprint=fingerprint,
        )
        missing = missing_chunk_files(cache_dir, [chunk.chunk_id for chunk in chunks])
        if missing:
            raise RuntimeError(
                f"{len(missing)} of {len(chunks)} RAG KV chunk files are missing under "
                f"{cache_dir} (first missing: {missing[0]}). Run: rag-jasper-bench "
                f"--precompute-kv-only --dataset-name {config.dataset_name} "
                f"--model {config.model} --chunk-size {config.chunk_size} "
                f"--context-window {config.context_window}"
            )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / "config.json", config.to_jsonable())
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / "system.json", system_metadata)

    from .embedder import build_cached_embedder
    from .retrieval import CorpusRetriever

    embedder = build_cached_embedder(config, mode="read")
    retriever = CorpusRetriever(config=config, chunks=chunks, embedder=embedder)
    setup_metrics: dict[str, Any] = {
        "dataset": config.dataset_name,
        "doc_count": len(docs),
        "chunk_count": len(chunks),
        "corpus_total_tokens": sum(chunk.token_count for chunk in chunks),
    }
    records: list[dict[str, Any]] = []
    client: Any | None = None
    try:
        setup_metrics.update(retriever.build())
        if config.answer_backend == "vllm-kv":
            from .answer_kv import RagKvAnswerClient

            assert cache_dir is not None and meta_base is not None
            client = RagKvAnswerClient(
                config,
                chunks=chunks,
                cache_dir=cache_dir,
                meta_base=meta_base,
                prompt_profile=spec.prompt_profile,
            )
        else:
            from .answer_prefix import RagPrefixAnswerClient

            client = RagPrefixAnswerClient(
                config,
                chunks_by_id=chunks_by_id,
                prompt_profile=spec.prompt_profile,
            )
        judge_client = None if config.skip_judge else build_judge_client(config)
        if not config.skip_judge and judge_client is None:
            raise RuntimeError(
                "Judge client is not configured. Use --skip-judge to write unjudged predictions."
            )
        client.start_llm()
        with JsonlWriter(config.run_dir / "predictions.jsonl") as writer:
            for index, query in enumerate(queries, start=1):
                record = _answer_one_query(
                    config,
                    query,
                    retriever=retriever,
                    client=client,
                    judge_client=judge_client,
                    chunk_text_by_id=chunk_text_by_id,
                    docs_by_id=docs_by_id,
                )
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
    finally:
        if client is not None:
            store_stats = getattr(client, "store_stats", None)
            if callable(store_stats):
                setup_metrics.update(store_stats())
            client.close()
        retriever.close()

    setup_metrics["run_wall_time_ms"] = (time.perf_counter() - started) * 1000
    summary = _write_rag_outputs(
        config,
        config.run_dir / "predictions.jsonl",
        records,
        saved_config=config.to_jsonable(),
        system_metadata=system_metadata,
        setup_metrics=setup_metrics,
        mode=config.result_mode(),
        write_reports=True,
    )
    write_json(config.run_dir / "setup.json", setup_metrics)
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


def judge_existing_run(config: RagBenchConfig) -> dict[str, Any]:
    predictions_path = config.run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    logger.info("Judging existing RAG run_id={} predictions={}", config.run_id, predictions_path)
    records = read_jsonl(predictions_path)
    saved_config = read_json_or_default(config.run_dir / "config.json", config.to_jsonable())
    saved_dataset = saved_config.get("dataset_name")
    if saved_dataset and saved_dataset != config.dataset_name:
        logger.warning(
            "Judging run_id={} recorded for dataset {}; adopting the saved dataset.",
            config.run_id,
            saved_dataset,
        )
    mode = str(saved_config.get("mode") or config.result_mode())
    system_metadata = read_json_or_default(config.run_dir / "system.json", {})
    setup_metrics = read_json_or_default(config.run_dir / "setup.json", {})

    judge_client = build_judge_client(config)
    if judge_client is None:
        raise RuntimeError("--judge-only requires --judge vllm.")

    judged_now = 0
    for row_number, record in enumerate(records, start=1):
        if not config.rejudge and is_judged(record):
            continue
        try:
            judge_payload = judge_rag_record(config, judge_client, record)
        except Exception as exc:
            record["judge"] = failed_judge_payload(exc, config)
            _write_rag_outputs(
                config,
                predictions_path,
                records,
                saved_config=saved_config,
                system_metadata=system_metadata,
                setup_metrics=setup_metrics,
                mode=mode,
                write_reports=False,
            )
            raise RuntimeError(
                f"Judge request failed for row {row_number}/{len(records)} {record_label(record)}. "
                f"Saved progress to {predictions_path}; fix or restart the judge server, then "
                f"rerun --judge-only. Original error: {type(exc).__name__}: {exc}"
            ) from exc
        record["judge"] = judge_payload
        judged_now += 1
        _write_rag_outputs(
            config,
            predictions_path,
            records,
            saved_config=saved_config,
            system_metadata=system_metadata,
            setup_metrics=setup_metrics,
            mode=mode,
            write_reports=False,
        )

    summary = _write_rag_outputs(
        config,
        predictions_path,
        records,
        saved_config=saved_config,
        system_metadata=system_metadata,
        setup_metrics=setup_metrics,
        mode=mode,
        write_reports=True,
    )
    logger.info(
        "Finished deferred judging run_id={} judged_now={} judged={} accuracy={}",
        config.run_id,
        judged_now,
        summary["judged_count"],
        format_accuracy(summary["metrics"].get("accuracy")),
    )
    for question_type, stats in (summary["metrics"].get("accuracy_by_type") or {}).items():
        logger.info(
            "Type {} accuracy={} ({}/{})",
            question_type,
            format_accuracy(stats.get("accuracy")),
            stats.get("correct"),
            stats.get("total"),
        )
    return summary


def _answer_one_query(
    config: RagBenchConfig,
    query: RagQuery,
    *,
    retriever: Any,
    client: Any,
    judge_client: Any,
    chunk_text_by_id: dict[str, str],
    docs_by_id: dict[str, RagDocument],
) -> dict[str, Any]:
    query_started = time.perf_counter()
    hits, retrieval_metrics = retriever.search(query.question, top_k=config.top_k)
    answer = client.answer(query, hits, query_started_at=query_started)
    answer.content = final_answer_text(answer.content)
    if config.skip_judge:
        judge_payload = skipped_judge_payload(config)
    else:
        judge_payload = judge_rag_answer(
            config,
            judge_client,
            question=query.question,
            gold_answers=query.gold_answers,
            predicted_answer=answer.content,
        )
    retrieval_quality = retrieval_metrics_for_query(query, hits, chunk_text_by_id)
    return build_query_record(
        config,
        query,
        hits,
        answer,
        retrieval_metrics=retrieval_metrics,
        retrieval_quality=retrieval_quality,
        judge_payload=judge_payload,
        docs_by_id=docs_by_id,
    )


def _write_rag_outputs(
    config: RagBenchConfig,
    predictions_path: Path,
    records: list[dict[str, Any]],
    *,
    saved_config: dict[str, Any],
    system_metadata: dict[str, Any],
    setup_metrics: dict[str, Any],
    mode: str,
    write_reports: bool,
) -> dict[str, Any]:
    replace_jsonl(predictions_path, records)
    summary = summarize_rag_records(
        records,
        run_id=config.run_id,
        mode=mode,
        config=saved_config,
        system_metadata=system_metadata,
        setup_metrics=setup_metrics,
    )
    write_json(config.run_dir / "summary.json", summary)
    if write_reports:
        write_csv(
            config.run_dir / "query_metrics.csv",
            build_query_metric_rows(records),
            QUERY_METRICS_COLUMNS,
        )
    return summary


def _kv_bytes_per_token(config: RagBenchConfig) -> int | None:
    dtype_bytes = _DTYPE_BYTES.get(config.kv_dtype.lower())
    if dtype_bytes is None:
        return None
    geometry = _KV_GEOMETRY.get(config.model)
    if geometry is None:
        geometry = _kv_geometry_from_autoconfig(config.model)
    if geometry is None:
        return None
    layers, kv_heads, head_dim = geometry
    return layers * kv_heads * head_dim * 2 * dtype_bytes


def _kv_geometry_from_autoconfig(model: str) -> tuple[int, int, int] | None:
    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model)
    except Exception:
        return None
    layers = getattr(hf_config, "num_hidden_layers", None)
    kv_heads = getattr(hf_config, "num_key_value_heads", None) or getattr(
        hf_config, "num_attention_heads", None
    )
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(hf_config, "hidden_size", None)
        heads = getattr(hf_config, "num_attention_heads", None)
        if hidden_size and heads:
            head_dim = hidden_size // heads
    if not layers or not kv_heads or not head_dim:
        return None
    return int(layers), int(kv_heads), int(head_dim)
