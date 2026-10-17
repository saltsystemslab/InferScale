"""Result rows of the throughput benchmark and the CSV, JSON, and Markdown reports built from them.

``ThroughputResultRow`` is the schema: its fields, in order, are the CSV columns
of every condition file and of the merged file, and their types drive how rows
read back from CSV are parsed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

from benchmarks.common.atomic_files import atomic_write_text
from benchmarks.common.files import write_csv, write_json
from benchmarks.memory.throughput.config import ALL_CONDITIONS, ThroughputConfig


@dataclass(frozen=True, slots=True)
class ThroughputResultRow:
    run_id: str
    model: str
    model_label: str
    condition: str
    vector_backend: str | None
    jasper_effective_beam_width: int | None
    num_users: int
    fact_count: float
    requests_per_user: int
    total_requests: int
    throughput_qps: float
    avg_latency_ms: float
    generation_time_s: float
    retrieval_time_s: float
    vector_search_time_s: float
    prompt_build_time_s: float
    kv_compose_time_s: float
    kv_verify_time_s: float
    memory_setup_time_s: float
    kv_precompute_time_s: float
    engine_startup_time_s: float
    kv_prefix_caching: int
    kv_store_gpu_mb: float
    kv_store_backend: str
    kv_store_host_mb: float
    kv_store_write_time_s: float
    kv_h2d_bytes: int
    kv_h2d_avg_ms: float
    kv_h2d_p95_ms: float
    kv_h2d_overlap_ratio: float
    kv_staging_stall_ms: float
    kv_requests_loaded: int
    total_input_tokens: int
    total_output_tokens: int
    input_tokens_per_second: float
    output_tokens_per_second: float


RESULT_COLUMNS = tuple(item.name for item in fields(ThroughputResultRow))


def _optional(parse: Callable[[Any], Any]) -> Callable[[Any], Any]:
    return lambda value: None if value in (None, "") else parse(value)


_PARSERS: dict[Any, Callable[[Any], Any]] = {
    str: str,
    int: int,
    float: float,
    str | None: _optional(str),
    int | None: _optional(int),
}
_COLUMN_PARSERS = {
    name: _PARSERS[hint] for name, hint in get_type_hints(ThroughputResultRow).items()
}


def build_result_row(
    config: ThroughputConfig,
    num_users: int,
    *,
    condition: str,
    vector_backend: str | None = None,
    jasper_effective_beam_width: int | None = None,
    fact_count: float = 0.0,
    generation_time_s: float,
    retrieval_time_s: float = 0.0,
    vector_search_time_s: float = 0.0,
    prompt_build_time_s: float = 0.0,
    memory_setup_time_s: float = 0.0,
    kv_precompute_time_s: float = 0.0,
    kv_compose_time_s: float = 0.0,
    kv_verify_time_s: float = 0.0,
    engine_startup_time_s: float = 0.0,
    kv_store_gpu_mb: float = 0.0,
    kv_store_backend: str = "gpu",
    kv_store_host_mb: float = 0.0,
    kv_store_write_time_s: float = 0.0,
    kv_h2d_bytes: int = 0,
    kv_h2d_avg_ms: float = 0.0,
    kv_h2d_p95_ms: float = 0.0,
    kv_h2d_overlap_ratio: float = 0.0,
    kv_staging_stall_ms: float = 0.0,
    kv_requests_loaded: int = 0,
    total_input_tokens: int,
    total_output_tokens: int,
) -> dict[str, Any]:
    """One measured user count as a result row; QPS counts only the generation time."""
    total_requests = num_users * config.requests_per_user
    if generation_time_s <= 0:
        raise RuntimeError("Measured benchmark time must be greater than zero.")
    row = ThroughputResultRow(
        run_id=config.run_id,
        model=config.model,
        model_label=config.model_label,
        condition=condition,
        vector_backend=vector_backend,
        jasper_effective_beam_width=jasper_effective_beam_width,
        num_users=num_users,
        fact_count=fact_count,
        requests_per_user=config.requests_per_user,
        total_requests=total_requests,
        throughput_qps=total_requests / generation_time_s,
        avg_latency_ms=generation_time_s / total_requests * 1000,
        generation_time_s=generation_time_s,
        retrieval_time_s=retrieval_time_s,
        vector_search_time_s=vector_search_time_s,
        prompt_build_time_s=prompt_build_time_s,
        kv_compose_time_s=kv_compose_time_s,
        kv_verify_time_s=kv_verify_time_s,
        memory_setup_time_s=memory_setup_time_s,
        kv_precompute_time_s=kv_precompute_time_s,
        engine_startup_time_s=engine_startup_time_s,
        kv_prefix_caching=int(config.kv_enable_prefix_caching),
        kv_store_gpu_mb=kv_store_gpu_mb,
        kv_store_backend=kv_store_backend,
        kv_store_host_mb=kv_store_host_mb,
        kv_store_write_time_s=kv_store_write_time_s,
        kv_h2d_bytes=kv_h2d_bytes,
        kv_h2d_avg_ms=kv_h2d_avg_ms,
        kv_h2d_p95_ms=kv_h2d_p95_ms,
        kv_h2d_overlap_ratio=kv_h2d_overlap_ratio,
        kv_staging_stall_ms=kv_staging_stall_ms,
        kv_requests_loaded=kv_requests_loaded,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        input_tokens_per_second=total_input_tokens / generation_time_s,
        output_tokens_per_second=total_output_tokens / generation_time_s,
    )
    return asdict(row)


def validate_result_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A row with every column present, parsed to the schema's types."""
    condition = str(row.get("condition"))
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown throughput condition: {condition}")
    return _coerce_row(row)


def _coerce_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse a row read from CSV or JSON into the typed values of the schema."""
    missing = [column for column in RESULT_COLUMNS if column not in row]
    if missing:
        raise ValueError("Throughput result is missing columns: " + ", ".join(missing))
    return {column: _COLUMN_PARSERS[column](row[column]) for column in RESULT_COLUMNS}


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
    existing: Iterable[dict[str, Any]], replacements: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Later rows replace earlier ones with the same condition and user count."""
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in [*existing, *replacements]:
        normalized = validate_result_row(row)
        by_key[(normalized["condition"], normalized["num_users"])] = normalized
    return sorted(by_key.values(), key=_result_sort_key)


def write_reports(
    config: ThroughputConfig, rows: Iterable[dict[str, Any]], *, system_metadata: dict[str, Any]
) -> dict[str, Any]:
    normalized = merge_result_rows([], rows)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    for condition in ALL_CONDITIONS:
        condition_rows = [row for row in normalized if row["condition"] == condition]
        if condition_rows:
            write_csv(
                condition_csv_path(config.run_dir, condition), condition_rows, list(RESULT_COLUMNS)
            )
    write_csv(config.run_dir / "throughput_merged.csv", normalized, list(RESULT_COLUMNS))
    summary = build_summary(config, normalized, system_metadata=system_metadata)
    write_json(config.run_dir / "summary.json", summary)
    atomic_write_text(
        config.run_dir / "throughput_report.md", render_markdown_report(config, normalized)
    )
    return summary


def build_summary(
    config: ThroughputConfig, rows: Iterable[dict[str, Any]], *, system_metadata: dict[str, Any]
) -> dict[str, Any]:
    values = list(rows)
    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in ALL_CONDITIONS:
        qps_values = [row["throughput_qps"] for row in values if row["condition"] == condition]
        if qps_values:
            condition_summaries[condition] = {
                "row_count": len(qps_values),
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
        "user_counts": list(config.user_counts),
        "config": config.to_jsonable(),
        "system": system_metadata,
    }


def render_markdown_report(config: ThroughputConfig, rows: Iterable[dict[str, Any]]) -> str:
    values = list(rows)
    lines = [
        "# Multi-User Throughput Report",
        "",
        f"Model: `{config.model}`.",
        "",
        f"Dataset: `{config.dataset_path}`; every user retrieves from the same LoCoMo "
        "conversation and asks that conversation's own questions.",
        "",
        "QPS for every condition times only the synchronous vLLM generation call over the "
        "prepared batch.",
        "",
        "Query embedding, vector retrieval, prompt construction, chunked-RoPE KV composition and "
        "registration, token-equivalence verification, memory setup, per-fact KV precomputation, "
        "and model startup are reported separately in the timing table and excluded from QPS.",
        "",
        *_qps_table(config, values),
        "",
        "## Timing Components",
        "",
        *_timing_table(values),
    ]
    return "\n".join(lines) + "\n"


def _qps_table(config: ThroughputConfig, rows: list[dict[str, Any]]) -> list[str]:
    by_count: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_count[row["num_users"]][row["condition"]] = row
    table_rows = []
    for count in config.user_counts:
        group = by_count.get(count, {})
        kv_qps = _qps(group.get("kv_injection"))
        mem0_jasper_qps = _qps(group.get("mem0_jasper"))
        fact_counts = [row["fact_count"] for row in group.values() if row["fact_count"] > 0]
        table_rows.append(
            (
                str(count),
                f"{fact_counts[0]:.0f}" if fact_counts else "-",
                _format_qps(_qps(group.get("mem0_qdrant"))),
                _format_qps(mem0_jasper_qps),
                _format_qps(kv_qps),
                _format_ratio(kv_qps, mem0_jasper_qps),
            )
        )
    headers = (
        "Users",
        "Facts/user",
        "Mem0 Qdrant QPS",
        "Mem0 Jasper QPS",
        "KV QPS",
        "KV / Mem0 Jasper",
    )
    return _markdown_table(headers, table_rows, align=("---:",) * len(headers))


def _timing_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = (
        "Condition",
        "Users",
        "Setup (s)",
        "Retrieval (s)",
        "KV compose (s)",
        "KV verify (s)",
        "Prompt build (s)",
        "Generation (s)",
    )
    table_rows = []
    for row in sorted(rows, key=_result_sort_key):
        setup = row["memory_setup_time_s"] + row["kv_precompute_time_s"]
        table_rows.append(
            (
                row["condition"],
                str(row["num_users"]),
                f"{setup:.3f}",
                f"{row['retrieval_time_s']:.3f}",
                f"{row['kv_compose_time_s']:.3f}",
                f"{row['kv_verify_time_s']:.3f}",
                f"{row['prompt_build_time_s']:.3f}",
                f"{row['generation_time_s']:.3f}",
            )
        )
    return _markdown_table(headers, table_rows, align=("---", *(("---:",) * (len(headers) - 1))))


def _markdown_table(
    headers: Sequence[str], rows: Iterable[Sequence[str]], *, align: Sequence[str]
) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _result_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    return row["num_users"], ALL_CONDITIONS.index(row["condition"])


def _qps(row: dict[str, Any] | None) -> float | None:
    return None if row is None else row["throughput_qps"]


def _format_qps(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _format_ratio(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return "-"
    return f"{numerator / denominator:.2f}x"
