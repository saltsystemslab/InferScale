from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from benchmarks.common.json_fields import (
    ConfigError,
    expand_path,
    load_json_object,
    optional,
    reject_unknown_keys,
)
from benchmarks.common.paths import project_root

WHERE = "setup"
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
    setup = load_json_object(config_path)
    reject_unknown_keys(setup, SETUP_KEYS, WHERE)
    root = project_root()
    dataset_path = optional(setup, "dataset_path", str, DEFAULT_DATASET_PATH, WHERE)
    extraction_plan = optional(
        setup, "extraction_launch_config", str, DEFAULT_EXTRACTION_LAUNCH_CONFIG, WHERE
    )
    exports = {
        "LOCOMO_DATASET_PATH": expand_path(dataset_path, root=root),
        "LOCOMO_DATASET_URL": optional(setup, "dataset_url", str, DEFAULT_DATASET_URL, WHERE),
        "FRESH_REMOTE_BUILD": int(_enabled(setup, "fresh_build", default=False)),
        "SKIP_SUBMODULE_INIT": int(not _enabled(setup, "initialize_submodules", default=True)),
        "SKIP_EXTRACTION": int(not _enabled(setup, "extract_facts", default=True)),
        "EXTRACTION_LAUNCH_CONFIG": expand_path(extraction_plan, root=root),
        "JASPER_VECTORS_PER_SEGMENT_LOG2": _segment_size_log2(setup),
    }
    return "\n".join(f"export {name}={shlex.quote(str(value))}" for name, value in exports.items())


def _enabled(setup: dict[str, Any], key: str, *, default: bool) -> bool:
    return optional(setup, key, bool, default, WHERE)


def _segment_size_log2(setup: dict[str, Any]) -> int:
    size_log2 = optional(
        setup,
        "jasper_vectors_per_segment_log2",
        int,
        DEFAULT_JASPER_VECTORS_PER_SEGMENT_LOG2,
        WHERE,
    )
    if not 1 <= size_log2 <= 31:
        raise ConfigError(
            "setup: jasper_vectors_per_segment_log2 must be an integer between 1 and 31."
        )
    return size_log2
