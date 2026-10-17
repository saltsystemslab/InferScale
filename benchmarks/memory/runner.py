"""Run a memory benchmark end to end, or judge a saved run's predictions."""

from __future__ import annotations

from typing import Any

from loguru import logger

from benchmarks.common.deferred_judging import judge_pending_records
from benchmarks.common.files import read_json_or_default, read_jsonl, replace_jsonl, write_json
from benchmarks.common.judge import failed_judge_payload, format_accuracy
from benchmarks.common.system import collect_system_metadata
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.judging import judge_metadata, judge_record, record_label
from benchmarks.memory.modes import result_mode
from benchmarks.memory.prediction import PREDICTIONS_FILE, run_prediction_mode
from benchmarks.memory.reporting import (
    read_sample_setup_report,
    write_query_reports,
    write_sample_setup_report,
)
from benchmarks.memory.results import summarize_records
from benchmarks.memory.runtime_clients import RuntimeClients, build_clients, judge_client_for

CONFIG_FILE = "config.json"
SYSTEM_FILE = "system.json"
SUMMARY_FILE = "summary.json"


def run_benchmark(config: MemoryRunConfig, clients: RuntimeClients | None = None) -> dict[str, Any]:
    logger.info(
        "Starting benchmark run_id={} dataset={} results_dir={}",
        config.run_id,
        config.dataset_path,
        config.run_dir,
    )
    config.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.run_dir / CONFIG_FILE, config.to_jsonable())
    owns_clients = clients is None
    runtime_clients = clients or build_clients(config)
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / SYSTEM_FILE, system_metadata)
    try:
        prediction = run_prediction_mode(config, runtime_clients)
    finally:
        if owns_clients:
            runtime_clients.answer_client.close()
    write_sample_setup_report(config.run_dir, prediction.sample_setup_metrics)
    summary = write_run_outputs(
        config,
        prediction.records,
        config_json=config.to_jsonable(),
        system_metadata=system_metadata,
        sample_setup_metrics=prediction.sample_setup_metrics,
        write_reports=True,
    )
    logger.info(
        "Finished benchmark run_id={} questions={} judged={} accuracy={}",
        config.run_id,
        summary["question_count"],
        summary["judged_count"],
        format_accuracy(summary.get("metrics", {}).get("accuracy")),
    )
    _log_category_accuracy(summary)
    logger.info("Wrote results to {}", config.run_dir)
    return summary


def judge_existing_run(config: MemoryRunConfig) -> dict[str, Any]:
    predictions_path = config.run_dir / PREDICTIONS_FILE
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")
    logger.info("Judging existing run_id={} predictions={}", config.run_id, predictions_path)
    records = read_jsonl(predictions_path)
    saved_config = read_json_or_default(config.run_dir / CONFIG_FILE, config.to_jsonable())
    saved_config = _reconcile_with_evidence(config, saved_config)
    system_metadata = read_json_or_default(config.run_dir / SYSTEM_FILE, {})
    sample_setup_metrics = read_sample_setup_report(config.run_dir)
    judge_client = judge_client_for(config)
    if judge_client is None:
        raise RuntimeError("Deferred judging requires the vllm judge provider.")

    def save(rows: list[dict[str, Any]], final: bool) -> dict[str, Any]:
        return write_run_outputs(
            config,
            rows,
            config_json=saved_config,
            system_metadata=system_metadata,
            sample_setup_metrics=sample_setup_metrics,
            write_reports=final,
        )

    summary, judged_now = judge_pending_records(
        records,
        predictions_path=predictions_path,
        rejudge=config.rejudge,
        judge=lambda record: judge_record(config, judge_client, record),
        failed_payload=lambda exc: failed_judge_payload(exc, judge_metadata(config)),
        save=save,
        label=record_label,
    )
    logger.info(
        "Finished deferred judging run_id={} judged_now={} judged={} accuracy={}",
        config.run_id,
        judged_now,
        summary["judged_count"],
        format_accuracy(summary.get("metrics", {}).get("accuracy")),
    )
    _log_category_accuracy(summary)
    return summary


def write_run_outputs(
    config: MemoryRunConfig,
    records: list[dict[str, Any]],
    *,
    config_json: dict[str, Any],
    system_metadata: dict[str, Any],
    sample_setup_metrics: list[dict[str, Any]],
    write_reports: bool,
) -> dict[str, Any]:
    """Write predictions.jsonl and summary.json, plus the query reports when asked."""
    replace_jsonl(config.run_dir / PREDICTIONS_FILE, records)
    summary = summarize_records(
        records,
        run_id=config.run_id,
        mode=result_mode(config),
        config=config_json,
        system_metadata=system_metadata,
        sample_setup_metrics=sample_setup_metrics,
    )
    write_json(config.run_dir / SUMMARY_FILE, summary)
    if write_reports:
        write_query_reports(config.run_dir, records)
    return summary


def _reconcile_with_evidence(
    config: MemoryRunConfig, saved_config: dict[str, Any]
) -> dict[str, Any]:
    """Keep deferred judging on the evidence protocol the run was recorded with.

    Without rejudge the run's saved with_evidence value is adopted, so the
    judge JSON cannot silently judge under a different protocol than
    config.json. With rejudge the judge JSON wins and the saved config.json
    is rewritten to match.
    """
    saved_with_evidence = bool(saved_config.get("with_evidence", False))
    if config.rejudge:
        if config.with_evidence != saved_with_evidence:
            logger.warning(
                "Rejudging run_id={} with with_evidence={} (run was recorded with {}); "
                "updating the saved config.json.",
                config.run_id,
                config.with_evidence,
                saved_with_evidence,
            )
            saved_config = {**saved_config, "with_evidence": config.with_evidence}
            write_json(config.run_dir / CONFIG_FILE, saved_config)
        return saved_config
    if config.with_evidence and not saved_with_evidence:
        raise RuntimeError(
            "judge.with_evidence=true conflicts with this run's saved configuration "
            "(with_evidence=false). Judging only the unjudged records under a different "
            "evidence protocol would mix protocols within one run; set rejudge=true to "
            "re-judge every record with evidence instead."
        )
    if saved_with_evidence and not config.with_evidence:
        logger.info(
            "Adopting the run's saved with_evidence=true for deferred judging of run_id={}.",
            config.run_id,
        )
        config.judge.with_evidence = True
    return saved_config


def _log_category_accuracy(summary: dict[str, Any]) -> None:
    by_category = summary.get("metrics", {}).get("accuracy_by_category") or {}
    for category, stats in by_category.items():
        logger.info(
            "Category {} ({}) accuracy={} ({}/{})",
            category,
            stats.get("name"),
            format_accuracy(stats.get("accuracy")),
            stats.get("correct"),
            stats.get("total"),
        )
