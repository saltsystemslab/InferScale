"""The LMCache CacheBlend arm's settings and how they reach vLLM.

Every LMCache setting travels through vLLM's ``kv_connector_extra_config``
under a ``lmcache.`` prefix, which LMCache's adapter applies on top of its
defaults; no environment variable or YAML file is involved.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from benchmarks.common.json_fields import (
    ConfigError,
    int_list,
    optional,
    reject_unknown_keys,
    require,
)
from inferscale.v1.config import EngineConfig

LMCACHE_VERSION_PIN = "0.4.4"
CONNECTOR_NAME = "BlendConnector"
CONNECTOR_MODULE = "benchmarks.common.lmcache.connector"
KV_ROLE = "kv_both"
# "package" runs the vLLM adapter shipped in the lmcache package; "vendored"
# runs the copy of that adapter inside vLLM.
ADAPTERS = ("package", "vendored")
PROMPT_LAYOUT = "header-sep-chunks-sep-footer-v1"
CONFIG_FILE_VARIABLE = "LMCACHE_CONFIG_FILE"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_BLEND_RECOMPUTE_RATIO = 0.15
DEFAULT_BLEND_CHECK_LAYERS = (1,)
DEFAULT_BLEND_SPECIAL_STR = " # # "
DEFAULT_ADAPTER = "package"
LMCACHE_KEYS = (
    "max_local_cpu_size_gb",
    "chunk_size",
    "blend_recompute_ratio",
    "blend_check_layers",
    "blend_special_str",
    "adapter",
)


@dataclass(slots=True, frozen=True)
class LMCacheConfig:
    """The ``lmcache`` section of a benchmark JSON file."""

    max_local_cpu_size_gb: float
    chunk_size: int = DEFAULT_CHUNK_SIZE
    blend_recompute_ratio: float = DEFAULT_BLEND_RECOMPUTE_RATIO
    blend_check_layers: tuple[int, ...] = DEFAULT_BLEND_CHECK_LAYERS
    blend_special_str: str = DEFAULT_BLEND_SPECIAL_STR
    adapter: str = DEFAULT_ADAPTER

    def to_jsonable(self) -> dict[str, Any]:
        """The section plus the facts the arm fixes, recorded in a run's config.json."""
        return {
            "max_local_cpu_size_gb": self.max_local_cpu_size_gb,
            "chunk_size": self.chunk_size,
            "blend_recompute_ratio": self.blend_recompute_ratio,
            "blend_check_layers": list(self.blend_check_layers),
            "blend_special_str": self.blend_special_str,
            "adapter": self.adapter,
            "connector": CONNECTOR_NAME,
            "connector_module": CONNECTOR_MODULE,
            "kv_role": KV_ROLE,
            "vllm_prefix_caching": False,
            "prompt_layout": PROMPT_LAYOUT,
            "version_pin": LMCACHE_VERSION_PIN,
        }


def parse_lmcache_config(section: Mapping[str, Any], where: str) -> LMCacheConfig:
    """Validate an ``lmcache`` section; the CPU cache size is always explicit."""
    reject_unknown_keys(section, LMCACHE_KEYS, where)
    size = require(section, "max_local_cpu_size_gb", (int, float), where)
    if not math.isfinite(size) or size <= 0:
        raise ConfigError(f"{where}.max_local_cpu_size_gb must be a finite positive number.")
    chunk_size = optional(section, "chunk_size", int, DEFAULT_CHUNK_SIZE, where)
    if chunk_size < 1:
        raise ConfigError(f"{where}.chunk_size must be >= 1.")
    ratio = optional(
        section, "blend_recompute_ratio", (int, float), DEFAULT_BLEND_RECOMPUTE_RATIO, where
    )
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ConfigError(f"{where}.blend_recompute_ratio must be between 0 and 1.")
    layers = int_list(section, "blend_check_layers", where, default=DEFAULT_BLEND_CHECK_LAYERS)
    if any(layer < 0 for layer in layers):
        raise ConfigError(f"{where}.blend_check_layers entries must be >= 0.")
    special_str = optional(section, "blend_special_str", str, DEFAULT_BLEND_SPECIAL_STR, where)
    if not special_str.strip():
        raise ConfigError(f"{where}.blend_special_str must contain a non-blank separator.")
    adapter = optional(section, "adapter", str, DEFAULT_ADAPTER, where)
    if adapter not in ADAPTERS:
        raise ConfigError(f"{where}.adapter must be one of {ADAPTERS}.")
    return LMCacheConfig(
        max_local_cpu_size_gb=float(size),
        chunk_size=chunk_size,
        blend_recompute_ratio=float(ratio),
        blend_check_layers=layers,
        blend_special_str=special_str,
        adapter=adapter,
    )


def connector_extra_config(config: LMCacheConfig) -> dict[str, Any]:
    """The ``kv_connector_extra_config`` the connector receives.

    Keys with the ``lmcache.`` prefix become LMCache engine settings. The bare
    ``use_layerwise`` is read by vLLM itself to run CUDA graphs piecewise, and
    ``use_native`` selects the adapter copy that ships inside vLLM.
    """
    extra: dict[str, Any] = {
        "lmcache.chunk_size": config.chunk_size,
        "lmcache.local_cpu": True,
        "lmcache.max_local_cpu_size": float(config.max_local_cpu_size_gb),
        "lmcache.use_layerwise": True,
        "lmcache.enable_blending": True,
        "lmcache.blend_recompute_ratios": [config.blend_recompute_ratio],
        "lmcache.blend_check_layers": list(config.blend_check_layers),
        "lmcache.blend_special_str": config.blend_special_str,
        "use_layerwise": True,
    }
    if config.adapter == "vendored":
        extra["use_native"] = True
    return extra


def lmcache_kv_transfer_config(config: LMCacheConfig) -> dict[str, Any]:
    """The ``kv_transfer_config`` that loads ``BlendConnector`` with these settings."""
    if os.environ.get(CONFIG_FILE_VARIABLE):
        raise RuntimeError(
            f"{CONFIG_FILE_VARIABLE} is set; an LMCache config file would override the "
            "benchmark's lmcache settings. Unset it and configure the lmcache JSON section."
        )
    return {
        "kv_connector": CONNECTOR_NAME,
        "kv_role": KV_ROLE,
        "kv_connector_module_path": CONNECTOR_MODULE,
        "kv_connector_extra_config": connector_extra_config(config),
    }


def engine_config_for_lmcache(engine: EngineConfig) -> EngineConfig:
    """The arm's engine settings: LMCache's blending replaces vLLM's prefix cache."""
    return replace(engine, enable_prefix_caching=False)
