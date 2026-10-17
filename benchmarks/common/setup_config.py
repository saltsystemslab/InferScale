"""Installation settings from ``configs/setup.json``, exported for scripts/setup_remote.sh."""

from __future__ import annotations

import shlex
from pathlib import Path

from benchmarks.common.json_fields import (
    ConfigError,
    expand_path,
    load_json_object,
    optional,
    reject_unknown_keys,
)
from benchmarks.common.paths import project_root

SETUP_KEYS = (
    "runtime_config",
    "dataset_path",
    "dataset_url",
    "fresh_build",
    "initialize_submodules",
    "extract_facts",
    "extraction_launch_config",
    "jasper_vectors_per_segment_log2",
)
DEFAULT_DATASET_PATH = "data/locomo10.json"
DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DEFAULT_EXTRACTION_LAUNCH_CONFIG = "configs/launch/memory-extract.json"
DEFAULT_JASPER_VECTORS_PER_SEGMENT_LOG2 = 12


def setup_shell_exports(config_path: Path) -> str:
    """Quoted ``export`` lines for the setup script's inputs."""
    data = load_json_object(config_path)
    reject_unknown_keys(data, SETUP_KEYS, "setup")
    root = project_root()
    dataset_path = optional(data, "dataset_path", str, DEFAULT_DATASET_PATH, "setup")
    extraction_plan = optional(
        data, "extraction_launch_config", str, DEFAULT_EXTRACTION_LAUNCH_CONFIG, "setup"
    )
    values = {
        "LOCOMO_DATASET_PATH": expand_path(dataset_path, root=root),
        "LOCOMO_DATASET_URL": optional(data, "dataset_url", str, DEFAULT_DATASET_URL, "setup"),
        "FRESH_REMOTE_BUILD": int(optional(data, "fresh_build", bool, False, "setup")),
        "SKIP_SUBMODULE_INIT": int(
            not optional(data, "initialize_submodules", bool, True, "setup")
        ),
        "SKIP_EXTRACTION": int(not optional(data, "extract_facts", bool, True, "setup")),
        "EXTRACTION_LAUNCH_CONFIG": expand_path(extraction_plan, root=root),
        "JASPER_VECTORS_PER_SEGMENT_LOG2": _segment_size_log2(data),
    }
    return "\n".join(f"export {name}={shlex.quote(str(value))}" for name, value in values.items())


def _segment_size_log2(data: dict) -> int:
    """Validate the exponent of Jasper's compile-time vectors-per-segment constant."""
    value = optional(
        data,
        "jasper_vectors_per_segment_log2",
        int,
        DEFAULT_JASPER_VECTORS_PER_SEGMENT_LOG2,
        "setup",
    )
    if isinstance(value, bool) or not 1 <= value <= 31:
        raise ConfigError(
            "setup: jasper_vectors_per_segment_log2 must be an integer between 1 and 31."
        )
    return value
