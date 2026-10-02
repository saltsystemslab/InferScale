"""Read installation settings from JSON for the remote setup script."""
from __future__ import annotations
from pathlib import Path
import shlex

from benchmarks.common.config import expand_path, load_json_object, optional, reject_unknown_keys
from benchmarks.common.paths import project_root


def shell_exports(config_path: Path) -> str:
    """Return the installation settings selected by a fixed JSON workflow."""
    data = load_json_object(config_path)
    reject_unknown_keys(data, ("runtime_config", "dataset_path", "dataset_url", "fresh_build", "initialize_submodules", "extract_facts", "extraction_launch_config"), "setup")
    values = {
        "LOCOMO_DATASET_PATH": str(expand_path(optional(data, "dataset_path", str, "data/locomo10.json", "setup"), root=project_root())),
        "LOCOMO_DATASET_URL": optional(data, "dataset_url", str, "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json", "setup"),
        "FRESH_REMOTE_BUILD": int(optional(data, "fresh_build", bool, False, "setup")),
        "SKIP_SUBMODULE_INIT": int(not optional(data, "initialize_submodules", bool, True, "setup")),
        "SKIP_EXTRACTION": int(not optional(data, "extract_facts", bool, True, "setup")),
        "EXTRACTION_LAUNCH_CONFIG": str(expand_path(optional(data, "extraction_launch_config", str, "configs/launch/memory-extract.json", "setup"), root=project_root())),
    }
    return "\n".join(f"export {name}={shlex.quote(str(value))}" for name, value in values.items())

