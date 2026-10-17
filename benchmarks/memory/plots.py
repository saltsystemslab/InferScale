"""Token-vs-latency and token-vs-accuracy plots of a memory run.

pandas and matplotlib are optional: without them the CSV reports are still
written and the plots are skipped with a warning.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from benchmarks.common.stats import as_number

FIGURE_SIZE = (8, 5)
FIGURE_DPI = 160
X_LABEL = "Input prompt tokens"


@dataclass(slots=True, frozen=True)
class LatencyPlot:
    file_name: str
    y_column: str
    title: str
    y_label: str


LATENCY_PLOTS = (
    LatencyPlot(
        "tokens_vs_ttft.png",
        "time_to_first_token_ms",
        "Input Tokens vs Engine TTFT",
        "Engine time to first token (ms)",
    ),
    LatencyPlot(
        "tokens_vs_query_to_first_token.png",
        "query_to_first_token_ms",
        "Input Tokens vs Query-To-First-Token Latency",
        "Query to first token (ms)",
    ),
    LatencyPlot(
        "tokens_vs_query_to_answer.png",
        "query_to_answer_ms",
        "Input Tokens vs Query-To-Answer Latency",
        "Query to answer (ms)",
    ),
)


def write_token_plots(
    plots_dir: Path, rows: list[dict[str, Any]], accuracy_bins: list[dict[str, Any]]
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("Skipping token plots because matplotlib is unavailable: {}", exc)
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    for plot in LATENCY_PLOTS:
        _write_latency_scatter(plt, plots_dir / plot.file_name, rows, plot)
    _write_accuracy_plot(
        plt,
        plots_dir / "tokens_vs_accuracy_binned.png",
        accuracy_bins,
        missing_x_count=_judged_rows_missing_tokens(rows),
    )


def _write_latency_scatter(
    plt: Any, output_path: Path, rows: Iterable[dict[str, Any]], plot: LatencyPlot
) -> None:
    points, missing_x_count = _latency_points(rows, plot.y_column)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    if points:
        ax.scatter([x for x, _ in points], [y for _, y in points], alpha=0.7, s=28)
    else:
        _draw_empty(ax, f"No rows with input tokens and {plot.y_column}")
    _note_missing(ax, missing_x_count, "row")
    ax.set_title(plot.title)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(plot.y_label)
    ax.grid(True, alpha=0.25)
    _save_figure(plt, fig, output_path)


def _write_accuracy_plot(
    plt: Any, output_path: Path, bins: list[dict[str, Any]], *, missing_x_count: int
) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    if bins:
        midpoints = [row["token_bin_midpoint"] for row in bins]
        accuracies = [row["accuracy"] for row in bins]
        ax.errorbar(
            midpoints,
            accuracies,
            yerr=[
                [row["accuracy"] - row["ci_lower"] for row in bins],
                [row["ci_upper"] - row["accuracy"] for row in bins],
            ],
            fmt="o-",
            capsize=4,
        )
        for row in bins:
            ax.annotate(
                f"n={int(row['judged_count'])}",
                (row["token_bin_midpoint"], row["accuracy"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
    else:
        _draw_empty(ax, "No judged rows available")
    _note_missing(ax, missing_x_count, "judged row")
    ax.set_title("Input Tokens vs Binned Accuracy")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("Judged accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    _save_figure(plt, fig, output_path)


def _latency_points(
    rows: Iterable[dict[str, Any]], y_column: str
) -> tuple[list[tuple[float, float]], int]:
    """(tokens, latency) pairs, and how many rows had the latency but no token count."""
    points = []
    missing_x_count = 0
    for row in rows:
        y_value = as_number(row.get(y_column))
        if y_value is None:
            continue
        x_value = as_number(row.get("total_prompt_tokens"))
        if x_value is None:
            missing_x_count += 1
            continue
        points.append((x_value, y_value))
    return points, missing_x_count


def _judged_rows_missing_tokens(rows: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if row.get("judge_correct") in {0, 1} and as_number(row.get("total_prompt_tokens")) is None
    )


def _draw_empty(ax: Any, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


def _note_missing(ax: Any, count: int, noun: str) -> None:
    if count:
        ax.text(
            0.02,
            0.96,
            f"x missing: {count} {noun}(s)",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )


def _save_figure(plt: Any, fig: Any, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)
