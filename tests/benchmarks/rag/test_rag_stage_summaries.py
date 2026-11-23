from __future__ import annotations

import importlib
import json
import os

import pytest

from benchmarks.rag import run

STAGE_FUNCTIONS = {"run": ("answer_run", "run_answer")}


@pytest.fixture
def run_stage(tmp_path, monkeypatch, rag_log):
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps({"storage": {"runtime_root": str(tmp_path)}}), encoding="utf-8"
    )
    configured_log_levels: list[str] = []
    monkeypatch.setattr(run, "configure_logging", configured_log_levels.append)
    monkeypatch.setattr(os, "environ", os.environ.copy())

    def run_with(stage: str, summary: dict, **settings) -> list[str]:
        module_name, function_name = STAGE_FUNCTIONS[stage]
        module = importlib.import_module(f"benchmarks.rag.{module_name}")
        monkeypatch.setattr(module, function_name, lambda config: summary)
        config_path = tmp_path / "rag.json"
        config_path.write_text(json.dumps({"run_id": "existing", **settings}), encoding="utf-8")
        run.main(config_path, runtime_path=runtime_path, stage=stage)
        assert {level for level, _ in rag_log} == {"INFO"}
        return [message for _, message in rag_log]

    return run_with


def test_run_stage_reports_where_results_are_and_the_unjudged_answer_quality(run_stage, tmp_path):
    summary = {
        "question_count": 4,
        "judged_count": 0,
        "metrics": {"accuracy": None, "exact_match": 0.25, "f1": 1 / 3},
    }

    assert run_stage("run", summary) == [
        f"wrote results to {tmp_path / 'results' / 'existing'}",
        "questions=4 judged=0/4 em=0.2500 f1=0.3333",
    ]
