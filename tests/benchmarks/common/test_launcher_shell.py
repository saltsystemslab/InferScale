from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    target = tmp_path / "repo $(printf keep-literal)"
    (target / "benchmarks/common").mkdir(parents=True)
    shutil.copytree(ROOT / "scripts", target / "scripts")
    for path in ("benchmarks/__init__.py", "benchmarks/common/__init__.py", "benchmarks/common/config.py", "benchmarks/common/paths.py", "benchmarks/common/environment.py"):
        shutil.copy2(ROOT / path, target / path)
    (target / "configs/launch").mkdir(parents=True)
    return target


def _runtime(repo: Path, name: str) -> Path:
    venv = repo / f"{name} venv"
    interpreter = venv / "bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'interpreter': sys.argv[0], 'args': sys.argv[1:], "
        "'code': sys.stdin.read(), 'runtime': os.environ.get('INFERSCALE_RUNTIME_CONFIG'), "
        "'venv': os.environ.get('VENV_DIR')}))\n"
    )
    interpreter.chmod(0o755)
    path = repo / f"{name} runtime.json"
    path.write_text(json.dumps({
        "storage": {"runtime_root": str(repo / f"{name} data")},
        "build": {"venv_dir": str(venv)},
    }))
    return path


def _run(repo: Path, script: str, args: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["BENCHMARK_ENV_FILE"] = str(repo / "absent.env")
    env.pop("INFERSCALE_RUNTIME_CONFIG", None)
    return subprocess.run(
        ["/bin/bash", str(repo / "scripts" / script), *args], cwd=repo.parent,
        env=env, text=True, capture_output=True,
    )


def test_fixed_plan_selects_runtime_and_interpreter_before_launch(repo: Path) -> None:
    runtime = _runtime(repo, "selected $(printf keep-literal)")
    plan = repo / "configs/launch/memory.json"
    plan.write_text(json.dumps({"family": "memory", "runtime_config": str(runtime), "configs": ["unused.json"], "stage": "run", "dry_run": True}))

    process = _run(repo, "full_run.sh")

    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["runtime"] == str(runtime)
    assert result["interpreter"] == str(repo / "selected $(printf keep-literal) venv/bin/python")
    assert result["args"] == ["-", str(plan)]
    assert "launch(Path(sys.argv[1]))" in result["code"]
    assert "argparse" not in result["code"]


def test_judge_plan_uses_saved_manifest_runtime(repo: Path) -> None:
    runtime = _runtime(repo, "saved")
    manifest = repo / "saved manifest.json"
    manifest.write_text(json.dumps({"family": "memory", "runtime_config": str(runtime), "run_configs": ["unused.json"]}))
    plan = repo / "configs/launch/memory-judge.json"
    plan.write_text(json.dumps({"family": "memory", "manifest": str(manifest), "stage": "judge", "dry_run": True}))

    process = _run(repo, "judge.sh")

    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["runtime"] == str(runtime)


def test_judge_plan_runtime_overrides_saved_runtime(repo: Path) -> None:
    saved = _runtime(repo, "saved")
    selected = _runtime(repo, "selected")
    manifest = repo / "saved manifest.json"
    manifest.write_text(json.dumps({"family": "memory", "runtime_config": str(saved), "run_configs": ["unused.json"]}))
    plan = repo / "configs/launch/memory-judge.json"
    plan.write_text(json.dumps({"family": "memory", "manifest": str(manifest), "runtime_config": str(selected), "stage": "judge"}))

    process = _run(repo, "judge.sh")

    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["runtime"] == str(selected)


@pytest.mark.parametrize("script", ["full_run.sh", "judge.sh", "rag/full_run.sh", "serve_vllm.sh", "setup_remote.sh", "load_env.sh"])
def test_scripts_reject_removed_arguments_before_any_work(repo: Path, script: str) -> None:
    process = _run(repo, script, ("--dry-run",))
    assert process.returncode == 2
    assert "takes no arguments" in process.stderr
    assert not list(repo.glob("* venv"))


def test_serve_uses_its_fixed_json_workflow(repo: Path) -> None:
    runtime = _runtime(repo, "judge")
    (repo / "configs/serve.json").write_text(json.dumps({"runtime_config": str(runtime), "dry_run": True}))

    process = _run(repo, "serve_vllm.sh")

    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result["runtime"] == str(runtime)
    assert result["args"] == ["-"]
    assert 'serve(Path("configs/serve.json"))' in result["code"]


def test_programmatic_serve_preview_reads_json_without_starting_a_server(tmp_path, monkeypatch, capsys):
    from benchmarks.common.config import RuntimeConfig
    from benchmarks.common import serve
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"storage": {"runtime_root": str(tmp_path / "data")}, "judge_server": {"base_url": "http://127.0.0.1:8123/v1", "model": "judge/model"}}))
    plan = tmp_path / "serve.json"
    plan.write_text(json.dumps({"runtime_config": str(runtime_path), "dry_run": True}))
    monkeypatch.setattr(RuntimeConfig, "apply_environment", lambda self: None)
    monkeypatch.setattr(serve.subprocess, "call", lambda *args: pytest.fail("Preview started a server"))

    assert serve.serve(plan) == 0
    assert "judge/model" in capsys.readouterr().out
    assert not (tmp_path / "data").exists()
