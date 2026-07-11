from __future__ import annotations

from locomo_jasper_bench.throughput.config import parse_args
from locomo_jasper_bench.throughput.runner import build_worker_command, worker_specs


def test_kv_workers_are_split_per_user_count() -> None:
    config, _ = parse_args(
        ["--model", "test/model", "--user-counts", "10,25", "--run-id", "run"]
    )

    specs = worker_specs(config)

    by_condition: dict[str, list[tuple[int, ...]]] = {}
    for spec in specs:
        by_condition.setdefault(spec.condition, []).append(spec.user_counts)
    assert by_condition["no_memory"] == [(10, 25)]
    assert by_condition["mem0_qdrant"] == [(10, 25)]
    assert by_condition["mem0_jasper"] == [(10, 25)]
    assert by_condition["kv_injection"] == [(10,), (25,)]


def test_worker_command_targets_the_worker_module() -> None:
    config, _ = parse_args(["--model", "test/model", "--user-counts", "10", "--run-id", "run"])
    spec = worker_specs(config)[0]

    command = build_worker_command(config, spec)

    assert "locomo_jasper_bench.throughput.worker" in command
    assert "--condition" in command
    assert spec.condition in command
    assert "--user-counts" in command
