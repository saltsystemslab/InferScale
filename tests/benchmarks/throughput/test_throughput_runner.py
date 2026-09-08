from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.memory.throughput.config import DEFAULT_USER_COUNTS, ThroughputConfig
from benchmarks.memory.throughput.runner import build_worker_job, run_throughput, worker_specs


def _config(tmp_path: Path) -> ThroughputConfig:
    return ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
        run_id="dry-run-test",
        conditions=("mem0_qdrant", "kv_injection"),
        user_counts=(2, 3),
    )


def test_worker_specs_isolate_each_kv_user_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    specs = worker_specs(config)

    assert [spec.condition for spec in specs] == ["mem0_qdrant", "kv_injection", "kv_injection"]
    assert specs[0].user_counts == (2, 3)
    assert specs[1].user_counts == (2,)
    assert specs[2].user_counts == (3,)
    assert build_worker_job(config, specs[0]) == {
        "config_path": str(config.run_dir / "config.json"),
        "condition": "mem0_qdrant",
        "user_counts": [2, 3],
        "output_path": str(config.run_dir / "worker-results/mem0_qdrant.json"),
    }


def test_default_plan_isolates_kv_workers_per_user_count(tmp_path: Path) -> None:
    config = ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
        run_id="full-plan-test",
    )

    specs = worker_specs(config)

    assert len(specs) == 2 + len(DEFAULT_USER_COUNTS)  # 2 single workers + 1 kv worker per user count
    assert sum(spec.condition == "kv_injection" for spec in specs) == len(DEFAULT_USER_COUNTS)
    single_worker_conditions = [
        spec.condition for spec in specs if spec.condition != "kv_injection"
    ]
    assert single_worker_conditions == ["mem0_qdrant", "mem0_jasper"]


def test_dry_run_prints_jobs_without_creating_run_directory(
    tmp_path: Path,
    capsys,
) -> None:
    config = _config(tmp_path)

    assert run_throughput(config, dry_run=True) is None
    output = capsys.readouterr().out

    jobs = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert [job["condition"] for job in jobs] == ["mem0_qdrant", "kv_injection", "kv_injection"]
    assert [job["user_counts"] for job in jobs] == [[2, 3], [2], [3]]
    assert not config.run_dir.exists()


def test_runner_sends_json_to_isolated_workers_and_collects_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmarks.memory.throughput import runner, worker
    from benchmarks.memory.throughput.reporting import build_result_row

    config = _config(tmp_path)
    monkeypatch.setattr(runner, "_validate_runtime_requirements", lambda config: None)
    monkeypatch.setattr(runner, "collect_system_metadata", lambda: {})
    monkeypatch.setattr(worker, "run_condition", lambda config, condition, counts: [
        build_result_row(
            config, count, condition=condition, generation_time_s=1.0,
            total_input_tokens=count * 20, total_output_tokens=count * 10,
        )
        for count in counts
    ])
    jobs = []

    def run(command, *, input, text, check, cwd, env):
        assert command == [sys.executable, "-m", "benchmarks.memory.throughput.worker"]
        assert text is check is True
        assert Path(cwd).is_dir()
        job = json.loads(input)
        jobs.append(job)
        worker.main(job)

    monkeypatch.setattr(runner.subprocess, "run", run)

    summary = run_throughput(config)

    assert summary["row_count"] == 4
    assert [job["user_counts"] for job in jobs] == [[2, 3], [2], [3]]
    assert all(Path(job["output_path"]).is_file() for job in jobs)
    assert (config.run_dir / "throughput_merged.csv").is_file()


def test_validate_existing_config_accepts_pre_change_config_json(tmp_path: Path) -> None:
    from benchmarks.memory.throughput.runner import _validate_existing_config

    config = _config(tmp_path)
    config.run_dir.mkdir(parents=True)
    legacy = config.to_jsonable()
    # A config.json written before these fields existed has no such keys;
    # the recorded behavior was prefix caching off and the GPU store.
    for key in (
        "kv_enable_prefix_caching",
        "kv_store_backend",
        "kv_staging_slots",
        "local_store_dir",
    ):
        legacy.pop(key)
    (config.run_dir / "config.json").write_text(json.dumps(legacy), encoding="utf-8")

    resumed = _config(tmp_path)
    resumed.kv_enable_prefix_caching = False
    resumed.kv_store_backend = "gpu"
    resumed.kv_staging_slots = 4
    _validate_existing_config(resumed)  # must not raise

    mismatched = _config(tmp_path)
    mismatched.kv_enable_prefix_caching = True
    try:
        _validate_existing_config(mismatched)
    except RuntimeError as exc:
        assert "kv_enable_prefix_caching" in str(exc)
    else:
        raise AssertionError("Expected a config mismatch error for prefix caching on")


@pytest.mark.parametrize("value", [False, True])
def test_validate_existing_config_ignores_removed_device_kv_selection_switch(
    tmp_path: Path, value: bool
) -> None:
    from benchmarks.memory.throughput.runner import _validate_existing_config

    config = _config(tmp_path)
    config.run_dir.mkdir(parents=True)
    legacy = config.to_jsonable()
    legacy["jasper_device_kv_selection"] = value
    (config.run_dir / "config.json").write_text(json.dumps(legacy), encoding="utf-8")

    _validate_existing_config(config)
