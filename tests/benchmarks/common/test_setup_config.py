from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest

from benchmarks.common.config import ConfigError
from benchmarks.common.setup_config import setup_shell_exports


def _setup_file(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "setup.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _exports(text: str) -> dict[str, str]:
    assignments = (shlex.split(line.removeprefix("export "))[0] for line in text.splitlines())
    return dict(assignment.split("=", 1) for assignment in assignments)


def test_setup_exports_use_defaults_for_omitted_keys(tmp_path: Path) -> None:
    exports = _exports(setup_shell_exports(_setup_file(tmp_path, {})))

    assert exports["LOCOMO_DATASET_PATH"].endswith("data/locomo10.json")
    assert exports["EXTRACTION_LAUNCH_CONFIG"].endswith("configs/launch/memory-extract.json")
    assert exports["FRESH_REMOTE_BUILD"] == "0"
    assert exports["SKIP_SUBMODULE_INIT"] == "0"
    assert exports["SKIP_EXTRACTION"] == "0"
    assert exports["JASPER_VECTORS_PER_SEGMENT_LOG2"] == "12"


def test_setup_exports_reflect_selected_values(tmp_path: Path) -> None:
    selected = {
        "fresh_build": True,
        "initialize_submodules": False,
        "extract_facts": False,
        "jasper_vectors_per_segment_log2": 15,
    }

    exports = _exports(setup_shell_exports(_setup_file(tmp_path, selected)))

    assert exports["FRESH_REMOTE_BUILD"] == "1"
    assert exports["SKIP_SUBMODULE_INIT"] == "1"
    assert exports["SKIP_EXTRACTION"] == "1"
    assert exports["JASPER_VECTORS_PER_SEGMENT_LOG2"] == "15"


@pytest.mark.parametrize("value", [0, 32, True, "12"])
def test_setup_rejects_invalid_segment_sizes(tmp_path: Path, value: object) -> None:
    with pytest.raises(ConfigError):
        setup_shell_exports(_setup_file(tmp_path, {"jasper_vectors_per_segment_log2": value}))


def test_setup_rejects_unknown_keys(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        setup_shell_exports(_setup_file(tmp_path, {"jasper_segment_log2": 12}))
