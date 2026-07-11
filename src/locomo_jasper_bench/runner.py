from __future__ import annotations

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
    runtime_clients = clients or build_clients(config)
    system_metadata = collect_system_metadata()
    write_json(config.run_dir / "system.json", system_metadata)

    try:
        prediction_result = run_prediction_mode(config, runtime_clients)
        records = prediction_result.records
        sample_setup_metrics = prediction_result.sample_setup_metrics
    finally:
        if owns_clients:
            close_answer_client = getattr(runtime_clients.answer_client, "close", None)
            if callable(close_answer_client):
                close_answer_client()

    summary = _write_benchmark_outputs(config, records, system_metadata, sample_setup_metrics)
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


def judge_existing_run(config: BenchmarkConfig) -> dict[str, Any]:
    predictions_path = config.run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    logger.info("Judging existing run_id={} predictions={}", config.run_id, predictions_path)
    records = read_jsonl(predictions_path)
    saved_config = read_json_or_default(config.run_dir / "config.json", config.to_jsonable())
    saved_config = _reconcile_with_evidence(config, saved_config)
    system_metadata = read_json_or_default(config.run_dir / "system.json", {})
    sample_setup_metrics = read_sample_setup_report(config.run_dir)

    judge_client = build_judge_client(config)
    if judge_client is None:
        raise RuntimeError("--judge-only requires --judge vllm.")

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
    _log_category_accuracy(summary)
    return summary


def _reconcile_with_evidence(
    config: BenchmarkConfig,
    saved_config: dict[str, Any],
) -> dict[str, Any]:
    """Keep deferred judging on the run's recorded evidence protocol.

    Without --rejudge the run's saved with_evidence value is adopted, so a
    forgotten flag cannot silently judge under a different protocol than
    config.json. With --rejudge the CLI value wins and the saved config.json is
    rewritten to match.
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
            saved_config = dict(saved_config)
            saved_config["with_evidence"] = config.with_evidence
            write_json(config.run_dir / "config.json", saved_config)
        return saved_config

    if config.with_evidence and not saved_with_evidence:
        raise RuntimeError(
            "--with-evidence conflicts with this run's saved configuration "
            "(with_evidence=false). Judging only the unjudged records under a "
            "different evidence protocol would mix protocols within one run; "
            "pass --rejudge to re-judge every record with evidence instead."
        )
    if saved_with_evidence and not config.with_evidence:
        logger.info(
            "Adopting the run's saved with_evidence=true for deferred judging of run_id={}.",
            config.run_id,
        )
        config.with_evidence = True
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
