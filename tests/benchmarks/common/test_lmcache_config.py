from __future__ import annotations

import pytest

from benchmarks.common.json_fields import ConfigError
from benchmarks.common.lmcache.config import (
    CONNECTOR_MODULE,
    CONNECTOR_NAME,
    LMCacheConfig,
    connector_extra_config,
    engine_config_for_lmcache,
    lmcache_kv_transfer_config,
    parse_lmcache_config,
)
from inferscale.v1.config import EngineConfig


def test_parse_fills_defaults_and_records_the_fixed_facts() -> None:
    config = parse_lmcache_config({"max_local_cpu_size_gb": 20}, "memory config.lmcache")

    assert config == LMCacheConfig(max_local_cpu_size_gb=20.0)
    jsonable = config.to_jsonable()
    assert jsonable["blend_check_layers"] == [1]
    assert jsonable["vllm_prefix_caching"] is False
    assert jsonable["connector"] == CONNECTOR_NAME
    assert jsonable["version_pin"] == "0.4.4"


def test_parse_reads_every_key() -> None:
    section = {
        "max_local_cpu_size_gb": 1.5,
        "chunk_size": 64,
        "blend_recompute_ratio": 0.2,
        "blend_check_layers": [0, 2],
        "blend_special_str": " | ",
        "adapter": "vendored",
    }

    expected = LMCacheConfig(1.5, 64, 0.2, (0, 2), " | ", "vendored")
    assert parse_lmcache_config(section, "x") == expected


@pytest.mark.parametrize(
    "section,match",
    [
        ({}, "max_local_cpu_size_gb"),
        ({"max_local_cpu_size_gb": 0}, "finite positive"),
        ({"max_local_cpu_size_gb": True}, "max_local_cpu_size_gb"),
        ({"max_local_cpu_size_gb": 1, "chunk_size": 0}, "chunk_size"),
        ({"max_local_cpu_size_gb": 1, "blend_recompute_ratio": 1.5}, "between 0 and 1"),
        ({"max_local_cpu_size_gb": 1, "blend_check_layers": []}, "blend_check_layers"),
        ({"max_local_cpu_size_gb": 1, "blend_check_layers": [-1]}, ">= 0"),
        ({"max_local_cpu_size_gb": 1, "blend_check_layers": [1, 1]}, "duplicates"),
        ({"max_local_cpu_size_gb": 1, "blend_check_layers": [True]}, "blend_check_layers"),
        ({"max_local_cpu_size_gb": 1, "blend_special_str": "   "}, "non-blank"),
        ({"max_local_cpu_size_gb": 1, "adapter": "mp"}, "adapter"),
        ({"max_local_cpu_size_gb": 1, "typo": 1}, "unknown keys"),
    ],
)
def test_invalid_sections_are_rejected(section: dict, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        parse_lmcache_config(section, "x")


def test_extra_config_carries_every_lmcache_setting() -> None:
    config = LMCacheConfig(
        max_local_cpu_size_gb=16,
        chunk_size=128,
        blend_recompute_ratio=0.1,
        blend_check_layers=(1, 3),
        blend_special_str=" # #",
    )

    assert connector_extra_config(config) == {
        "lmcache.chunk_size": 128,
        "lmcache.local_cpu": True,
        "lmcache.max_local_cpu_size": 16.0,
        "lmcache.use_layerwise": True,
        "lmcache.enable_blending": True,
        "lmcache.blend_recompute_ratios": [0.1],
        "lmcache.blend_check_layers": [1, 3],
        "lmcache.blend_special_str": " # #",
        "use_layerwise": True,
    }
    assert connector_extra_config(LMCacheConfig(1.0, adapter="vendored"))["use_native"] is True


def test_kv_transfer_config_loads_the_blend_connector_by_module_path(monkeypatch) -> None:
    monkeypatch.delenv("LMCACHE_CONFIG_FILE", raising=False)

    transfer = lmcache_kv_transfer_config(LMCacheConfig(max_local_cpu_size_gb=2))

    assert transfer["kv_connector"] == CONNECTOR_NAME == "BlendConnector"
    assert transfer["kv_connector_module_path"] == CONNECTOR_MODULE
    assert transfer["kv_role"] == "kv_both"
    assert transfer["kv_connector_extra_config"]["lmcache.enable_blending"] is True


def test_an_lmcache_config_file_in_the_environment_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(tmp_path / "lmcache.yaml"))

    with pytest.raises(RuntimeError, match="LMCACHE_CONFIG_FILE"):
        lmcache_kv_transfer_config(LMCacheConfig(max_local_cpu_size_gb=2))


def test_engine_config_for_lmcache_only_turns_prefix_caching_off() -> None:
    engine = EngineConfig(
        gpu_memory_utilization=0.3,
        max_model_len=4096,
        block_size=32,
        enable_prefix_caching=True,
        connector_module="x.y",
    )

    assert engine_config_for_lmcache(engine) == EngineConfig(0.3, 4096, 32, False, "x.y")
    assert engine.enable_prefix_caching is True
