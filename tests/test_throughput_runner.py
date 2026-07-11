from __future__ import annotations

from pathlib import Path

from locomo_jasper_bench.throughput.config import ThroughputConfig
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
        user_counts=(2, 3),
    )


def test_worker_specs_isolate_each_kv_user_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    specs = worker_specs(config)

    assert [spec.condition for spec in specs] == ["no_memory", "kv_injection", "kv_injection"]
    assert specs[0].user_counts == (2, 3)
    assert specs[1].user_counts == (2,)
    assert specs[2].user_counts == (3,)
    assert "locomo_jasper_bench.throughput.worker" in build_worker_command(config, specs[0])
    assert "--user-counts" in build_worker_command(config, specs[0])


def test_default_plan_isolates_kv_workers_per_user_count(tmp_path: Path) -> None:
    config = ThroughputConfig(
        model="test/model",
        model_label="test",
        results_dir=tmp_path,
        run_id="full-plan-test",
    )

    specs = worker_specs(config)

    assert len(specs) == 7  # 3 single workers + 4 kv user counts
    assert sum(spec.condition == "kv_injection" for spec in specs) == 4
    single_worker_conditions = [
        spec.condition for spec in specs if spec.condition != "kv_injection"
    ]
    assert single_worker_conditions == ["no_memory", "mem0_qdrant", "mem0_jasper"]


def test_dry_run_prints_commands_without_creating_run_directory(
    tmp_path: Path,
    capsys,
) -> None:
    config = _config(tmp_path)

    assert run_throughput(config, dry_run=True) is None
    output = capsys.readouterr().out

    assert "--condition no_memory" in output
    assert output.count("--condition kv_injection") == 2
    assert "--user-counts 2,3" in output
    assert not config.run_dir.exists()
