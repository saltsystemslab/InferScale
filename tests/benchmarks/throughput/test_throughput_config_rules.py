from __future__ import annotations

from pathlib import Path

import pytest
from throughput_config import make_throughput_config

from benchmarks.common.config import RuntimeConfig
from benchmarks.memory.throughput.config import (
    ALL_CONDITIONS,
    DEFAULT_CONDITIONS,
    ThroughputConfig,
    condition_vector_backend,
    jasper_effective_beam_width,
)


def test_default_conditions_are_every_condition_that_needs_no_section_of_its_own() -> None:
    without_the_lmcache_arm = tuple(
        condition for condition in ALL_CONDITIONS if condition != "lmcache_blend"
    )

    assert without_the_lmcache_arm == DEFAULT_CONDITIONS


def test_each_cache_condition_retrieves_like_its_prompt_injection_counterpart() -> None:
    assert condition_vector_backend("kv_injection") == condition_vector_backend("mem0_jasper")
    assert condition_vector_backend("lmcache_blend") == condition_vector_backend("mem0_qdrant")
    with pytest.raises(KeyError):
        condition_vector_backend("prompt_injection")


def test_sweep_with_null_conditions_keeps_the_default_conditions_and_needs_no_lmcache(
    tmp_path: Path,
) -> None:
    runtime = RuntimeConfig.from_dict({"models": {"llama": "local/llama-test"}}, root=tmp_path)
    settings = {"model": "llama", "inferscale": {}, "sweeps": {"defaults": {"conditions": None}}}

    assert ThroughputConfig.from_dict(settings, runtime=runtime).conditions == DEFAULT_CONDITIONS


def test_run_and_scratch_directories_are_named_after_the_run_id(tmp_path: Path) -> None:
    config = make_throughput_config(tmp_path, run_id="named-run")

    assert config.run_dir == tmp_path / "throughput" / "named-run"
    assert config.local_store_scratch_dir == (
        tmp_path / "scratch" / "inferscale-bench-stores" / "named-run"
    )


def test_jasper_searches_at_least_as_wide_as_the_retrieval_depth(tmp_path: Path) -> None:
    narrow = make_throughput_config(tmp_path, jasper_beam_width=32, top_k=200)
    wide = make_throughput_config(tmp_path, jasper_beam_width=300, top_k=200)

    assert jasper_effective_beam_width(narrow) == 200
    assert jasper_effective_beam_width(wide) == 300
