"""The files a RAG run writes: predictions, the summary, and the per-query metrics table."""

from __future__ import annotations

from typing import Any

from benchmarks.common.files import replace_jsonl, write_csv, write_json
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.results import (
    QUERY_METRICS_COLUMNS,
    build_query_metric_rows,
    summarize_rag_records,
)

PREDICTIONS_FILE = "predictions.jsonl"
CONFIG_FILE = "config.json"
SYSTEM_FILE = "system.json"
SETUP_FILE = "setup.json"
SUMMARY_FILE = "summary.json"
QUERY_METRICS_FILE = "query_metrics.csv"


def write_run_outputs(
    config: RagBenchConfig,
    records: list[dict[str, Any]],
    *,
    saved_config: dict[str, Any],
    system_metadata: dict[str, Any],
    setup_metrics: dict[str, Any],
    write_reports: bool,
) -> dict[str, Any]:
    """Write predictions.jsonl and summary.json, plus query_metrics.csv when asked."""
    replace_jsonl(config.run_dir / PREDICTIONS_FILE, records)
    summary = summarize_rag_records(
        records,
        run_id=config.run_id,
        mode=config.result_mode(),
        config=saved_config,
        system_metadata=system_metadata,
        setup_metrics=setup_metrics,
    )
    write_json(config.run_dir / SUMMARY_FILE, summary)
    if write_reports:
        write_csv(
            config.run_dir / QUERY_METRICS_FILE,
            build_query_metric_rows(records),
            QUERY_METRICS_COLUMNS,
        )
    return summary
