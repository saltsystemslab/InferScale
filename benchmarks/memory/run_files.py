from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks.common.files import replace_jsonl, write_json
from benchmarks.memory.config import BenchmarkConfig
from benchmarks.memory.modes import existing_run_mode
from benchmarks.memory.reporting import write_query_reports
from benchmarks.memory.results import summarize_records


def write_deferred_judging_outputs(
    config: BenchmarkConfig,
    predictions_path: Path,
    records: list[dict[str, Any]],
    *,
    saved_config: dict[str, Any],
    system_metadata: dict[str, Any],
    sample_setup_metrics: list[dict[str, Any]] | None = None,
    write_reports: bool = True,
) -> dict[str, Any]:
    replace_jsonl(predictions_path, records)
    summary = summarize_records(
        records,
        run_id=config.run_id,
        mode=existing_run_mode(saved_config, records, config),
        config=saved_config,
        system_metadata=system_metadata,
        sample_setup_metrics=sample_setup_metrics,
    )
    write_json(config.run_dir / "summary.json", summary)
    if write_reports:
        write_query_reports(config.run_dir, records)
    return summary
