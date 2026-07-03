from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

QUERY_METRICS_COLUMNS = [
    "run_id",
    "mode",
    "sample_id",
    "question_id",
    "category",
    "memory_tokens",
    "query_tokens",
    "total_prompt_tokens",
    "time_to_first_token_ms",
    "query_to_first_token_ms",
    "query_to_answer_ms",
    "judge_correct",
    "retrieved_count",
    "selected_turn_count",
]

ACCURACY_BIN_COLUMNS = [
    "token_bin_start",
    "token_bin_end",
    "token_bin_midpoint",
    "judged_count",
    "correct_count",
    "accuracy",
    "ci_lower",
    "ci_upper",
]

SAMPLE_SETUP_COLUMNS = [
    "run_id",
    "mode",
    "sample_id",
    "question_count",
    "turn_count",
    "vector_backend",
    "memory_create_time_ms",
    "embedding_memory_build_time_ms",
    "vector_index_build_time_ms",
    "memory_setup_time_ms",
    "kv_precompute_time_ms",
    "answer_prepare_sample_time_ms",
    "sample_setup_time_ms",
]

SAMPLE_SETUP_NUMERIC_COLUMNS = set(SAMPLE_SETUP_COLUMNS) - {"run_id", "mode", "sample_id", "vector_backend"}


def write_query_reports(run_dir: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write per-query token metrics and token-vs-latency/accuracy plots."""
    run_path = Path(run_dir)
    rows = query_metric_rows(records)
    write_csv(run_path / "query_metrics.csv", rows, QUERY_METRICS_COLUMNS)

    plots_dir = run_path / "plots"
    accuracy_bins = accuracy_bin_rows(rows)
    write_csv(plots_dir / "tokens_vs_accuracy_binned.csv", accuracy_bins, ACCURACY_BIN_COLUMNS)

    try:
        pd, plt = _load_plotting()
    except ImportError as exc:
        logger.warning("Skipping token plots because pandas/matplotlib is unavailable: {}", exc)
        return

    df = pd.DataFrame(rows, columns=QUERY_METRICS_COLUMNS)
    _write_latency_scatter(
        df,
        plt,
        output_path=plots_dir / "tokens_vs_ttft.png",
        y_column="time_to_first_token_ms",
        title="Input Tokens vs Engine TTFT",
        y_label="Engine time to first token (ms)",
    )
    _write_latency_scatter(
        df,
        plt,
        output_path=plots_dir / "tokens_vs_query_to_first_token.png",
        y_column="query_to_first_token_ms",
        title="Input Tokens vs Query-To-First-Token Latency",
        y_label="Query to first token (ms)",
    )
    _write_latency_scatter(
        df,
        plt,
        output_path=plots_dir / "tokens_vs_query_to_answer.png",
        y_column="query_to_answer_ms",
        title="Input Tokens vs Query-To-Answer Latency",
        y_label="Query to answer (ms)",
    )
    _write_accuracy_binned_plot(
        pd.DataFrame(accuracy_bins, columns=ACCURACY_BIN_COLUMNS),
        plt,
        plots_dir / "tokens_vs_accuracy_binned.png",
        missing_x_count=_judged_missing_x_count(rows),
    )


def write_sample_setup_report(run_dir: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_csv(Path(run_dir) / "sample_setup_metrics.csv", rows, SAMPLE_SETUP_COLUMNS)


def read_sample_setup_report(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "sample_setup_metrics.csv"
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    key: _csv_read_number(value) if key in SAMPLE_SETUP_NUMERIC_COLUMNS else value
                    for key, value in row.items()
                }
            )
    return rows


def query_metric_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        memory_tokens = _number(metrics.get("kv_memory_tokens"))
        query_tokens = _number(metrics.get("kv_query_tokens"))
        total_prompt_tokens = None
        if memory_tokens is not None and query_tokens is not None:
            total_prompt_tokens = memory_tokens + query_tokens

        judge = record.get("judge")
        judge_correct = None
        if isinstance(judge, dict):
            correct = judge.get("correct")
            if correct is True:
                judge_correct = 1
            elif correct is False:
                judge_correct = 0

        retrieved = record.get("retrieved_memories")
        selected_turn_ids = metrics.get("kv_selected_turn_ids")
        rows.append(
            {
                "run_id": record.get("run_id"),
                "mode": record.get("mode"),
                "sample_id": record.get("sample_id"),
                "question_id": record.get("question_id"),
                "category": record.get("category"),
                "memory_tokens": memory_tokens,
                "query_tokens": query_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "time_to_first_token_ms": _number(metrics.get("time_to_first_token_ms")),
                "query_to_first_token_ms": _number(metrics.get("query_to_first_token_ms")),
                "query_to_answer_ms": _number(metrics.get("query_to_answer_ms")),
                "judge_correct": judge_correct,
                "retrieved_count": len(retrieved) if isinstance(retrieved, list) else None,
                "selected_turn_count": len(selected_turn_ids) if isinstance(selected_turn_ids, list) else None,
            }
        )
    return rows


def accuracy_bin_rows(rows: Iterable[dict[str, Any]], *, max_bins: int = 10) -> list[dict[str, Any]]:
    judged = [
        row
        for row in rows
        if _token_axis_value(row) is not None and row.get("judge_correct") in {0, 1}
    ]
    judged.sort(key=lambda row: float(_token_axis_value(row)))
    if not judged:
        return []

    bin_count = min(max_bins, max(1, round(math.sqrt(len(judged)))))
    chunk_size = max(1, math.ceil(len(judged) / bin_count))
    bins = []
    for start in range(0, len(judged), chunk_size):
        group = judged[start : start + chunk_size]
        tokens = [float(_token_axis_value(row)) for row in group]
        correct_count = sum(int(row["judge_correct"]) for row in group)
        count = len(group)
        accuracy = correct_count / count
        ci_lower, ci_upper = _wilson_interval(correct_count, count)
        bins.append(
            {
                "token_bin_start": min(tokens),
                "token_bin_end": max(tokens),
                "token_bin_midpoint": (min(tokens) + max(tokens)) / 2,
                "judged_count": count,
                "correct_count": correct_count,
                "accuracy": accuracy,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )
    return bins


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _load_plotting() -> tuple[Any, Any]:
    import pandas as pd

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return pd, plt


def _write_latency_scatter(
    df: Any,
    plt: Any,
    *,
    output_path: Path,
    y_column: str,
    title: str,
    y_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_rows, missing_x_count = _latency_plot_rows(df.to_dict("records"), y_column)
    data = df.__class__(plot_rows, columns=["total_prompt_tokens", y_column])
    if data.empty:
        _draw_empty_plot(ax, f"No rows with input tokens and {y_column}")
    else:
        ax.scatter(data["total_prompt_tokens"], data[y_column], alpha=0.7, s=28)
    if missing_x_count:
        ax.text(
            0.02,
            0.96,
            f"x missing: {missing_x_count} row(s)",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
    ax.set_title(title)
    ax.set_xlabel("Input prompt tokens")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_accuracy_binned_plot(
    bin_df: Any,
    plt: Any,
    output_path: Path,
    *,
    missing_x_count: int = 0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if bin_df.empty:
        _draw_empty_plot(ax, "No judged rows available")
    else:
        yerr_lower = bin_df["accuracy"] - bin_df["ci_lower"]
        yerr_upper = bin_df["ci_upper"] - bin_df["accuracy"]
        ax.errorbar(
            bin_df["token_bin_midpoint"],
            bin_df["accuracy"],
            yerr=[yerr_lower, yerr_upper],
            fmt="o-",
            capsize=4,
        )
        for _, row in bin_df.iterrows():
            ax.annotate(
                f"n={int(row['judged_count'])}",
                (row["token_bin_midpoint"], row["accuracy"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
    if missing_x_count:
        ax.text(
            0.02,
            0.96,
            f"x missing: {missing_x_count} judged row(s)",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )
    ax.set_title("Input Tokens vs Binned Accuracy")
    ax.set_xlabel("Input prompt tokens")
    ax.set_ylabel("Judged accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _draw_empty_plot(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _latency_plot_rows(rows: Iterable[dict[str, Any]], y_column: str) -> tuple[list[dict[str, float]], int]:
    plotted = []
    missing_x_count = 0
    for row in rows:
        y_value = _number(row.get(y_column))
        if y_value is None:
            continue
        x_value = _number(row.get("total_prompt_tokens"))
        if x_value is None:
            missing_x_count += 1
            continue
        plotted.append({"total_prompt_tokens": x_value, y_column: y_value})

    return plotted, missing_x_count


def _token_axis_value(row: dict[str, Any]) -> Any:
    return _number(row.get("total_prompt_tokens"))


def _judged_missing_x_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("judge_correct") in {0, 1} and _token_axis_value(row) is None
    )


def _wilson_interval(correct: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _csv_read_number(value: Any) -> Any:
    if value == "":
        return None
    number = _number(value)
    return number
