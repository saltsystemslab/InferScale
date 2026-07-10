from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..results import write_json
from .config import ALL_CONDITIONS, BenchmarkPoint, ThroughputConfig

RESULT_COLUMNS = (
    "run_id",
    "model",
    "model_label",
    "condition",
    "vector_backend",
    "jasper_effective_beam_width",
    "num_users",
    "memory_tokens",
    "requests_per_user",
    "total_requests",
    "wall_time_s",
    "throughput_qps",
    "avg_latency_ms",
    "generation_time_s",
    "retrieval_time_s",
    "vector_search_time_s",
    "prompt_build_time_s",
    "memory_setup_time_s",
    "kv_precompute_time_s",
    "engine_startup_time_s",
    "kv_store_gpu_mb",
    "total_input_tokens",
    "total_output_tokens",
    "input_tokens_per_second",
    "output_tokens_per_second",
)


def condition_csv_path(run_dir: str | Path, condition: str) -> Path:
    return Path(run_dir) / f"throughput_{condition}.csv"


def read_existing_results(run_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in ALL_CONDITIONS:
        path = condition_csv_path(run_dir, condition)
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(_coerce_row(row) for row in csv.DictReader(handle))
    return rows


def merge_result_rows(
    existing: Iterable[dict[str, Any]],
    replacements: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in [*existing, *replacements]:
        normalized = validate_result_row(row)
        key = (
            str(normalized["condition"]),
            int(normalized["num_users"]),
            int(normalized["memory_tokens"]),
        )
        by_key[key] = normalized
    return sorted(by_key.values(), key=_result_sort_key)


def write_reports(
    config: ThroughputConfig,
    rows: Iterable[dict[str, Any]],
    *,
    system_metadata: dict[str, Any],
) -> dict[str, Any]:
    normalized = merge_result_rows([], rows)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    for condition in ALL_CONDITIONS:
        condition_rows = [row for row in normalized if row["condition"] == condition]
        if condition_rows:
            _write_csv(condition_csv_path(config.run_dir, condition), condition_rows)
    _write_csv(config.run_dir / "throughput_merged.csv", normalized)
    summary = build_summary(config, normalized, system_metadata=system_metadata)
    write_json(config.run_dir / "summary.json", summary)
    _write_text(config.run_dir / "throughput_report.md", render_markdown_report(config, normalized))
    return summary


def build_summary(
    config: ThroughputConfig,
    rows: Iterable[dict[str, Any]],
    *,
    system_metadata: dict[str, Any],
) -> dict[str, Any]:
    values = list(rows)
    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in ALL_CONDITIONS:
        condition_rows = [row for row in values if row["condition"] == condition]
        if not condition_rows:
            continue
        qps_values = [float(row["throughput_qps"]) for row in condition_rows]
        condition_summaries[condition] = {
            "row_count": len(condition_rows),
            "average_qps": sum(qps_values) / len(qps_values),
            "minimum_qps": min(qps_values),
            "maximum_qps": max(qps_values),
        }
    return {
        "run_id": config.run_id,
        "benchmark": "multi-user-throughput",
        "model": config.model,
        "model_label": config.model_label,
        "row_count": len(values),
        "conditions": condition_summaries,
        "matrix": [
            {"num_users": point.num_users, "memory_tokens": point.memory_tokens}
            for point in config.matrix
        ],
        "config": config.to_jsonable(),
        "system": system_metadata,
    }


def render_markdown_report(config: ThroughputConfig, rows: Iterable[dict[str, Any]]) -> str:
    values = list(rows)
    by_point: dict[BenchmarkPoint, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in values:
        point = BenchmarkPoint(int(row["num_users"]), int(row["memory_tokens"]))
        by_point[point][str(row["condition"])] = row

    lines = [
        "# Multi-User Throughput Report",
        "",
        f"Model: `{config.model}`.",
        "",
        "QPS for no-memory, prompt-injection, and KV-injection rows times only the synchronous vLLM generation call.",
        "",
        "Mem0 QPS includes query embedding, vector retrieval, retrieval prompt construction, and vLLM generation.",
        "",
        "Memory setup, KV precomputation, and model startup are reported separately and excluded from QPS.",
        "",
        "KV generation timing includes the connector's exact token-prefix scan used to identify offline multi-user requests.",
        "",
        "| Users | Memory tokens | No memory QPS | Prompt QPS | KV QPS | Mem0 QPS | KV / Prompt | KV / Mem0 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in config.matrix:
        group = by_point.get(point, {})
        kv_qps = _qps(group.get("kv_injection"))
        prompt_qps = _qps(group.get("prompt_injection"))
        mem0_qps = _qps(group.get("mem0"))
        lines.append(
            "| "
            + " | ".join(
                (
                    str(point.num_users),
                    str(point.memory_tokens),
                    _format_qps(_qps(group.get("no_memory"))),
                    _format_qps(prompt_qps),
                    _format_qps(kv_qps),
                    _format_qps(mem0_qps),
                    _format_ratio(kv_qps, prompt_qps),
                    _format_ratio(kv_qps, mem0_qps),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Timing Components",
            "",
            "| Condition | Users | Memory tokens | Setup (s) | Retrieval (s) | Prompt build (s) | Generation (s) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(values, key=_result_sort_key):
        setup = float(row.get("memory_setup_time_s") or 0) + float(row.get("kv_precompute_time_s") or 0)
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["condition"]),
                    str(row["num_users"]),
                    str(row["memory_tokens"]),
                    f"{setup:.3f}",
                    f"{float(row.get('retrieval_time_s') or 0):.3f}",
                    f"{float(row.get('prompt_build_time_s') or 0):.3f}",
                    f"{float(row.get('generation_time_s') or 0):.3f}",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def validate_result_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = [column for column in RESULT_COLUMNS if column not in row]
    if missing:
        raise ValueError("Throughput result is missing columns: " + ", ".join(missing))
    condition = str(row["condition"])
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown throughput condition: {condition}")
    normalized = {column: row.get(column) for column in RESULT_COLUMNS}
    return _coerce_row(normalized)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in RESULT_COLUMNS})
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if normalized.get("vector_backend") == "":
        normalized["vector_backend"] = None
    for column in (
        "num_users",
        "memory_tokens",
        "requests_per_user",
        "total_requests",
        "total_input_tokens",
        "total_output_tokens",
    ):
        normalized[column] = int(normalized[column] or 0)
    beam_width = normalized.get("jasper_effective_beam_width")
    normalized["jasper_effective_beam_width"] = (
        None if beam_width in (None, "") else int(beam_width)
    )
    for column in (
        "wall_time_s",
        "throughput_qps",
        "avg_latency_ms",
        "generation_time_s",
        "retrieval_time_s",
        "vector_search_time_s",
        "prompt_build_time_s",
        "memory_setup_time_s",
        "kv_precompute_time_s",
        "engine_startup_time_s",
        "kv_store_gpu_mb",
        "input_tokens_per_second",
        "output_tokens_per_second",
    ):
        value = normalized.get(column)
        normalized[column] = None if value in (None, "") else float(value)
    return normalized


def _result_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    condition_order = ALL_CONDITIONS.index(str(row["condition"]))
    return int(row["num_users"]), int(row["memory_tokens"]), condition_order


def _qps(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    return float(row["throughput_qps"])


def _format_qps(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _format_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "-"
    return f"{numerator / denominator:.2f}x"
