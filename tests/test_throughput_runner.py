from __future__ import annotations

from pathlib import Path

from locomo_jasper_bench.throughput.config import BenchmarkPoint, ThroughputConfig
from locomo_jasper_bench.throughput.runner import (
    build_worker_command,
    run_throughput,
    worker_specs,
)


def _config(tmp_path: Path) -> ThroughputConfig:
    return ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
        run_id="dry-run-test",
        conditions=("no_memory", "kv_injection"),
        matrix=(BenchmarkPoint(2, 512), BenchmarkPoint(3, 1024)),
    )


def test_worker_specs_isolate_each_kv_point(tmp_path: Path) -> None:
    config = _config(tmp_path)
    specs = worker_specs(config)

    assert [spec.condition for spec in specs] == ["no_memory", "kv_injection", "kv_injection"]
    assert specs[0].matrix == config.matrix
    assert specs[1].matrix == (BenchmarkPoint(2, 512),)
    assert "locomo_jasper_bench.throughput.worker" in build_worker_command(config, specs[0])


def test_default_plan_has_fifteen_isolated_kv_workers(tmp_path: Path) -> None:
    config = ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
        run_id="full-plan-test",
    )

    specs = worker_specs(config)

    assert len(specs) == 18
    assert sum(spec.condition == "kv_injection" for spec in specs) == 15


def test_dry_run_prints_commands_without_creating_run_directory(
    tmp_path: Path,
    capsys,
) -> None:
    config = _config(tmp_path)

    assert run_throughput(config, dry_run=True) is None
    output = capsys.readouterr().out

    assert "--condition no_memory" in output
    assert output.count("--condition kv_injection") == 2
    assert not config.run_dir.exists()
