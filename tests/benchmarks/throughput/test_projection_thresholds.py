from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from throughput_config import make_throughput_config

from benchmarks.common.json_fields import load_json_object
from benchmarks.common.paths import project_root
from benchmarks.common.setup_config import DEFAULT_JASPER_VECTORS_PER_SEGMENT_LOG2
from benchmarks.memory.throughput import projection
from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.projection import (
    CorpusFootprint,
    check_kv_gpu_projection,
    check_pinned_host_projection,
)

GIB = 2**30
DEVICE_BYTES = 100 * GIB
NO_KV = CorpusFootprint(fact_count=1, token_count=1, kv_bytes=0, scaffold_token_count=0)
HALF_GIB_PER_REQUEST = CorpusFootprint(
    fact_count=1, token_count=1, kv_bytes=GIB // 2, scaffold_token_count=0
)


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    queried_devices: list[int] = []

    def get_device_properties(index: int) -> Any:
        queried_devices.append(index)
        return SimpleNamespace(total_memory=DEVICE_BYTES)

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=get_device_properties))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return queried_devices


def half_utilized(tmp_path: Path) -> ThroughputConfig:
    return make_throughput_config(tmp_path, kv_gpu_memory_utilization=0.5)


def check(config: ThroughputConfig, footprint: CorpusFootprint, **workload: int) -> None:
    check_kv_gpu_projection(
        config,
        **{"num_users": 1, "total_requests": 2, "unique_sample_count": 1, **workload},
        footprint=footprint,
    )


def test_each_sample_is_projected_one_jasper_graph_segment_at_the_measured_bytes_per_slot(
    tmp_path: Path, device: list[int]
) -> None:
    config = half_utilized(tmp_path)

    check(config, NO_KV, unique_sample_count=3564)
    with pytest.raises(RuntimeError, match="in the generate phase: .* graphs=47.0GiB"):
        check(config, NO_KV, unique_sample_count=3565)

    assert device == [0, 0]


def test_projected_graph_segment_has_the_vector_slots_that_setup_compiles_into_jasper() -> None:
    setup = load_json_object(project_root() / "configs/setup.json")
    configured_slots = 1 << setup["jasper_vectors_per_segment_log2"]
    default_slots = 1 << DEFAULT_JASPER_VECTORS_PER_SEGMENT_LOG2

    assert configured_slots == default_slots == projection.JASPER_SEGMENT_SLOTS


def test_projection_leaves_three_percent_of_the_device_to_the_cuda_context_and_fragmentation(
    tmp_path: Path, device: list[int]
) -> None:
    config = half_utilized(tmp_path)

    check(config, HALF_GIB_PER_REQUEST, total_requests=93, unique_sample_count=0)
    with pytest.raises(RuntimeError, match="in the generate phase: projected=97.5GiB"):
        check(config, HALF_GIB_PER_REQUEST, total_requests=95, unique_sample_count=0)


def test_unswappable_pinned_corpus_leaves_a_tenth_of_the_host_memory_for_the_engine_started_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_throughput_config(tmp_path)
    monkeypatch.setattr(projection, "available_host_memory_bytes", lambda: 10 * GIB)

    check_pinned_host_projection(config, corpus_bytes=8.9 * GIB, num_users=3, total_requests=6)
    with pytest.raises(RuntimeError) as failure:
        check_pinned_host_projection(config, corpus_bytes=9.1 * GIB, num_users=3, total_requests=6)

    assert str(failure.value) == (
        "Projected pinned-host KV corpus exceeds available RAM: "
        "corpus=9.1GiB available=10.0GiB (users=3, requests=6). "
        "Reduce the corpus or free host memory."
    )


def test_projection_can_be_imported_on_a_host_without_loading_torch() -> None:
    root = project_root()
    probe = "import sys, benchmarks.memory.throughput.projection; print('torch' in sys.modules)"

    imported = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(root), str(root / "src")])},
    )

    assert imported.stdout.strip() == "False"
