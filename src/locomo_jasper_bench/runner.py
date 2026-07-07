from __future__ import annotations

from dataclasses import replace
from typing import Any

from loguru import logger

from .clients_factory import RuntimeClients, build_clients, build_judge_client
from .config import BenchmarkConfig
from .judging import (
    failed_judge_payload,
    format_accuracy,
    is_judged,
    judge_record,
    record_label,
)
from .modes import result_mode
from .openai_batch_judging import (
    OpenAIBatchJudgeError,
    OpenAIResponsesBatchJudgeClient,
)
from .prediction import run_prediction_mode
from .reporting import read_sample_setup_report, write_query_reports, write_sample_setup_report
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
    prediction_config = _prediction_config(config)
    runtime_clients = clients or build_clients(prediction_config)
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / "system.json", system_metadata)

    try:
        prediction_result = run_prediction_mode(prediction_config, runtime_clients)
        records = prediction_result.records
        sample_setup_metrics = prediction_result.sample_setup_metrics
    finally:
        if owns_clients:
            close_answer_client = getattr(runtime_clients.answer_client, "close", None)
            if callable(close_answer_client):
                close_answer_client()

    if _uses_openai_batch_judge(config):
        try:
            _judge_openai_batch(config, records)
        except OpenAIBatchJudgeError as exc:
            _write_benchmark_outputs(config, records, system_metadata, sample_setup_metrics)
            raise RuntimeError(
                f"OpenAI batch judge failed for run_id={config.run_id}. "
                f"Saved progress to {config.run_dir / 'predictions.jsonl'}. "
                f"Original error: {exc}"
            ) from exc

    summary = _write_benchmark_outputs(config, records, system_metadata, sample_setup_metrics)
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
    sample_setup_metrics = read_sample_setup_report(config.run_dir)

    if config.judge_provider == "openai":
        try:
            batch_result = _judge_openai_batch(config, records)
        except OpenAIBatchJudgeError as exc:
            write_deferred_judging_outputs(
                config,
                predictions_path,
                records,
                saved_config=saved_config,
                system_metadata=system_metadata,
                sample_setup_metrics=sample_setup_metrics,
                write_reports=True,
            )
            raise RuntimeError(
                f"OpenAI batch judge failed for run_id={config.run_id}. "
                f"Saved progress to {predictions_path}. Original error: {exc}"
            ) from exc
        summary = write_deferred_judging_outputs(
            config,
            predictions_path,
            records,
            saved_config=saved_config,
            system_metadata=system_metadata,
            sample_setup_metrics=sample_setup_metrics,
            write_reports=True,
        )
        logger.info(
            "Finished deferred OpenAI batch judging run_id={} submitted={} judged={} accuracy={}",
            config.run_id,
            batch_result.submitted_count,
            summary["judged_count"],
            format_accuracy(summary.get("metrics", {}).get("accuracy")),
        )
        return summary

    judge_client = build_judge_client(config)
    if judge_client is None:
        raise RuntimeError("--judge-only requires --judge vllm or --judge openai.")

    judged_now = 0
    for row_number, record in enumerate(records, start=1):
        if not config.rejudge and is_judged(record):
            continue
        try:
            judge_payload = judge_record(config, judge_client, record)
        except Exception as exc:
            record["judge"] = failed_judge_payload(exc, config)
            write_deferred_judging_outputs(
                config,
                predictions_path,
                records,
                saved_config=saved_config,
                system_metadata=system_metadata,
                sample_setup_metrics=sample_setup_metrics,
                write_reports=False,
            )
            raise RuntimeError(
                f"Judge request failed for row {row_number}/{len(records)} {record_label(record)}. "
                f"Saved progress to {predictions_path}; fix or restart the judge server, then rerun --judge-only. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        record["judge"] = judge_payload
        judged_now += 1
        write_deferred_judging_outputs(
            config,
            predictions_path,
            records,
            saved_config=saved_config,
            system_metadata=system_metadata,
            sample_setup_metrics=sample_setup_metrics,
            write_reports=False,
        )

    summary = write_deferred_judging_outputs(
        config,
        predictions_path,
        records,
        saved_config=saved_config,
        system_metadata=system_metadata,
        sample_setup_metrics=sample_setup_metrics,
        write_reports=True,
    )
    logger.info(
        "Finished deferred judging run_id={} judged_now={} judged={} accuracy={}",
        config.run_id,
        judged_now,
        summary["judged_count"],
        format_accuracy(summary.get("metrics", {}).get("accuracy")),
    )
    return summary


def _prediction_config(config: BenchmarkConfig) -> BenchmarkConfig:
    if _uses_openai_batch_judge(config):
        return replace(config, skip_judge=True)
    return config


def _uses_openai_batch_judge(config: BenchmarkConfig) -> bool:
    return config.judge_provider == "openai" and not config.skip_judge


def _judge_openai_batch(config: BenchmarkConfig, records: list[dict[str, Any]]):
    if not config.judge_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to use --judge openai.")
    client = OpenAIResponsesBatchJudgeClient(
        api_key=config.judge_api_key,
        base_url=config.judge_base_url,
    )
    return client.judge_records(config, records)


def _write_benchmark_outputs(
    config: BenchmarkConfig,
    records: list[dict[str, Any]],
    system_metadata: dict[str, Any],
    sample_setup_metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    replace_jsonl(config.run_dir / "predictions.jsonl", records)
    summary = summarize_records(
        records,
        run_id=config.run_id,
        mode=result_mode(config),
        config=config.to_jsonable(),
        system_metadata=system_metadata,
        sample_setup_metrics=sample_setup_metrics,
    )
    write_json(config.run_dir / "summary.json", summary)
    write_sample_setup_report(config.run_dir, sample_setup_metrics)
    write_query_reports(config.run_dir, records)
    return summary
