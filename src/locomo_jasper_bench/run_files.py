from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .modes import existing_run_mode
from .reporting import write_query_reports
from .results import JsonlWriter, summarize_records, write_json


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def replace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with JsonlWriter(tmp_path) as writer:
        for row in rows:
            writer.write(row)
    tmp_path.replace(path)


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


def read_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return default
    return data
