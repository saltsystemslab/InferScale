from __future__ import annotations

from typing import Any

from loguru import logger

from .clients import OpenAICompatibleChatClient
from .clients_factory import RuntimeClients, build_clients
from .config import BenchmarkConfig
from .evaluation import QuestionEvaluator
from .judging import (
    failed_judge_payload,
    format_accuracy,
    is_judged,
    judge_record,
    record_label,
)
from .modes import result_mode
from .prediction import (
    PreparedQuestion,
    PreparedSample,
    planned_question_count,
    run_kv_prediction_mode,
    run_prediction_mode,
    should_log_progress,
)
from .results import summarize_records, write_json
from .run_files import read_json_or_default, read_jsonl, replace_jsonl, write_deferred_judging_outputs
from .system import collect_system_metadata


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
        records = run_prediction_mode(config, runtime_clients)
    finally:
        if owns_clients:
            close_answer_client = getattr(runtime_clients.answer_client, "close", None)
            if callable(close_answer_client):
                close_answer_client()

    summary = summarize_records(
        records,
        run_id=config.run_id,
        mode=result_mode(config),
        config=config.to_jsonable(),
        system_metadata=system_metadata,
    )
    write_json(config.run_dir / "summary.json", summary)
    logger.info(
        "Finished benchmark run_id={} questions={} judged={} accuracy={}",
        config.run_id,
        summary["question_count"],
        summary["judged_count"],
        format_accuracy(summary.get("metrics", {}).get("accuracy")),
    )
    logger.info("Wrote results to {}", config.run_dir)
    return summary


def judge_existing_run(config: BenchmarkConfig) -> dict[str, Any]:
    predictions_path = config.run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    logger.info("Judging existing run_id={} predictions={}", config.run_id, predictions_path)
    records = read_jsonl(predictions_path)
    saved_config = read_json_or_default(config.run_dir / "config.json", config.to_jsonable())
    system_metadata = read_json_or_default(config.run_dir / "system.json", {})
    judge_client = OpenAICompatibleChatClient(
        base_url=config.judge_base_url,
        api_key=config.judge_api_key,
        model=config.judge_model,
    )

    judged_now = 0
    summary: dict[str, Any] | None = None
    for row_number, record in enumerate(records, start=1):
        if is_judged(record):
            continue
        try:
            judge_payload = judge_record(config, judge_client, record)
        except Exception as exc:
            record["judge"] = failed_judge_payload(exc)
            write_deferred_judging_outputs(
                config,
                predictions_path,
                records,
                saved_config=saved_config,
                system_metadata=system_metadata,
            )
            raise RuntimeError(
                f"Judge request failed for row {row_number}/{len(records)} {record_label(record)}. "
                f"Saved progress to {predictions_path}; fix or restart the judge server, then rerun --judge-only. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        record["judge"] = judge_payload
        judged_now += 1
        summary = write_deferred_judging_outputs(
            config,
            predictions_path,
            records,
            saved_config=saved_config,
            system_metadata=system_metadata,
        )

    if summary is None:
        summary = write_deferred_judging_outputs(
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
        format_accuracy(summary.get("metrics", {}).get("accuracy")),
    )
    return summary


# Backward-compatible private aliases for tests and ad hoc scripts that imported
# helpers from runner.py before the cleanup.
_run_prediction_mode = run_prediction_mode
_run_kv_prediction_mode = run_kv_prediction_mode
_planned_question_count = planned_question_count
_should_log_progress = should_log_progress
_result_mode = result_mode
_format_accuracy = format_accuracy
_failed_judge_payload = failed_judge_payload
_judge_record = judge_record
_is_judged = is_judged
_record_label = record_label
_read_jsonl = read_jsonl
_replace_jsonl = replace_jsonl
_write_deferred_judging_outputs = write_deferred_judging_outputs
_read_json_or_default = read_json_or_default
