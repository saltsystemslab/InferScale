from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .results import write_json


def compare_run_summaries(paths: list[str | Path]) -> dict[str, Any]:
    rows = []
    for path in paths:
        summary_path = _summary_path(Path(path))
        summary = _read_summary(summary_path)
        rows.append(_summary_row(summary, summary_path))
    return {
        "run_count": len(rows),
        "runs": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="locomo-compare-runs",
        description="Compare benchmark run summary.json files.",
        allow_abbrev=False,
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or summary.json files.")
    parser.add_argument("--json-output", type=Path, help="Optional path for machine-readable comparison JSON.")
    args = parser.parse_args(argv)

    comparison = compare_run_summaries(args.runs)
    print(_markdown_table(comparison["runs"]))
    if args.json_output is not None:
        write_json(args.json_output, comparison)
        print(f"wrote comparison JSON to {args.json_output}")


def _summary_path(path: Path) -> Path:
    if path.is_dir():
        return path / "summary.json"
    return path


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Summary file must contain a JSON object: {path}")
    return data


def _summary_row(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    config = summary.get("config")
    if not isinstance(config, dict):
        config = {}
    retrieval_diagnostics = metrics.get("retrieval_diagnostics")
    if not isinstance(retrieval_diagnostics, dict):
        retrieval_diagnostics = {}

    return {
        "run_id": summary.get("run_id") or summary_path.parent.name,
        "summary_path": str(summary_path),
        "backend": config.get("vector_backend"),
        "judge_model": config.get("judge_model"),
        "jasper_beam_width": config.get("jasper_beam_width"),
        "vector_normalize": bool(config.get("vector_normalize", False)),
        "accuracy": metrics.get("accuracy"),
        "judged_count": summary.get("judged_count"),
        "question_count": summary.get("question_count"),
        "vector_db_query_time_ms_avg": _numeric_summary_value(metrics, "vector_db_query_time_ms", "avg"),
        "vector_db_queries_per_sec": metrics.get("vector_db_queries_per_sec"),
        "exact_recall_at_requested_top_k_avg": _retrieval_summary_value(
            retrieval_diagnostics,
            "exact_recall_at_requested_top_k",
            "avg",
        ),
        "exact_top_k_answer_accuracy": metrics.get("exact_top_k_answer_accuracy"),
    }


def _numeric_summary_value(metrics: dict[str, Any], metric_key: str, summary_key: str) -> Any:
    summary = metrics.get(metric_key)
    if not isinstance(summary, dict):
        return None
    return summary.get(summary_key)


def _retrieval_summary_value(retrieval_diagnostics: dict[str, Any], metric_key: str, summary_key: str) -> Any:
    summary = retrieval_diagnostics.get(metric_key)
    if not isinstance(summary, dict):
        return None
    return summary.get(summary_key)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "run_id",
        "judge",
        "beam",
        "norm",
        "accuracy",
        "judged",
        "query_ms_avg",
        "qps",
        "exact_recall",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [
            str(row.get("run_id") or ""),
            str(row.get("judge_model") or ""),
            str(row.get("jasper_beam_width") or ""),
            "yes" if row.get("vector_normalize") else "no",
            _format_optional_float(row.get("accuracy")),
            _format_count_pair(row.get("judged_count"), row.get("question_count")),
            _format_optional_float(row.get("vector_db_query_time_ms_avg")),
            _format_optional_float(row.get("vector_db_queries_per_sec")),
            _format_optional_float(row.get("exact_recall_at_requested_top_k_avg")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_count_pair(numerator: Any, denominator: Any) -> str:
    if numerator is None or denominator is None:
        return ""
    return f"{numerator}/{denominator}"


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
