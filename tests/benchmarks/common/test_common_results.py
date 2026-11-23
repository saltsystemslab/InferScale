from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.common.answers import final_answer_text
from benchmarks.common.files import write_json
from benchmarks.common.stats import numeric_summary


@pytest.mark.parametrize(
    "content,answer",
    [
        ("plain answer", "plain answer"),
        ("ANSWER: first ANSWER: last ", "last"),
    ],
)
def test_final_answer_is_the_text_after_the_last_answer_marker(content, answer):
    assert final_answer_text(content) == answer


def test_json_report_with_a_number_json_cannot_hold_keeps_the_existing_file(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text("original")

    with pytest.raises(ValueError, match="not JSON compliant"):
        write_json(report, {"metric": float("nan")})

    assert report.read_text() == "original"


def test_numeric_summary_orders_its_samples_and_skips_missing_ones():
    assert numeric_summary(sample for sample in (5, None, 1, 4, 2, 3)) == {
        "count": 5,
        "avg": 3.0,
        "min": 1.0,
        "p50": 3.0,
        "p95": 4.8,
        "max": 5.0,
    }
    assert list(numeric_summary([])) == ["count", "avg", "min", "p50", "p95", "max"]
    assert numeric_summary([None]) == {
        "count": 0,
        "avg": None,
        "min": None,
        "p50": None,
        "p95": None,
        "max": None,
    }
