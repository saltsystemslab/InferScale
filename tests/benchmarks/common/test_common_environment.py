from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig
from benchmarks.common.environment import runtime_shell_exports

ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP_MODULES = (
    "benchmarks.common.environment",
    "benchmarks.common.setup_config",
)


def test_rejected_environment_name_leaves_no_runtime_folders_behind(tmp_path: Path):
    runtime = RuntimeConfig.from_dict(
        {"storage": {"runtime_root": str(tmp_path / "runtime")}, "environment": {"BAD-NAME": "x"}},
        root=tmp_path,
    )

    with pytest.raises(ConfigError, match="Invalid runtime environment name: 'BAD-NAME'"):
        runtime_shell_exports(runtime, prepare=True)

    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize("module", BOOTSTRAP_MODULES)
def test_modules_the_setup_scripts_import_need_only_the_standard_library(module):
    without_site_packages = subprocess.run(
        [sys.executable, "-S", "-c", f"import {module}"],
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )

    assert without_site_packages.returncode == 0, without_site_packages.stderr
