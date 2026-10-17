"""The judge stage: judge the unjudged predictions of a saved run and rewrite its summary."""

from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.common.deferred_judging import judge_pending_records
from benchmarks.common.files import read_json_or_default, read_jsonl
from benchmarks.common.judge import failed_judge_payload, format_accuracy
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.judging import judge_client_for, judge_metadata, judge_rag_record, record_label
from benchmarks.rag.run_outputs import (
    CONFIG_FILE,
    PREDICTIONS_FILE,
    SETUP_FILE,
    SYSTEM_FILE,
    write_run_outputs,
)


def judge_existing_run(config: RagBenchConfig) -> dict[str, Any]:
    predictions_path = config.run_dir / PREDICTIONS_FILE
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")
    logger.info("Judging existing RAG run_id={} predictions={}", config.run_id, predictions_path)
    records = read_jsonl(predictions_path)
    saved_config = _saved_config(config)
    system_metadata = read_json_or_default(config.run_dir / SYSTEM_FILE, {})
    setup_metrics = read_json_or_default(config.run_dir / SETUP_FILE, {})
    judge_client = judge_client_for(config)
    if judge_client is None:
        raise RuntimeError("Deferred judging requires the vllm judge provider.")

    def save(rows: list[dict[str, Any]], final: bool) -> dict[str, Any]:
        return write_run_outputs(
            config,
            rows,
            saved_config=saved_config,
            system_metadata=system_metadata,
            setup_metrics=setup_metrics,
            write_reports=final,
        )

    summary, judged_now = judge_pending_records(
        records,
        predictions_path=predictions_path,
        rejudge=config.rejudge,
        judge=lambda record: judge_rag_record(config, judge_client, record),
        failed_payload=lambda exc: failed_judge_payload(exc, judge_metadata(config)),
        save=save,
        label=record_label,
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


def _saved_config(config: RagBenchConfig) -> dict[str, Any]:
    saved = read_json_or_default(config.run_dir / CONFIG_FILE, config.to_jsonable())
    saved_dataset = saved.get("dataset_name")
    if saved_dataset and saved_dataset != config.dataset_name:
        logger.warning(
            "Judging run_id={} recorded for dataset {}; adopting the saved dataset.",
            config.run_id,
            saved_dataset,
        )
    return saved
