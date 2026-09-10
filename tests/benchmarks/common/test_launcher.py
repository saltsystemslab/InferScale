from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig, load_json_object
from benchmarks.common import launcher

ROOT = Path(__file__).resolve().parents[3]
STAMP = "20260908T120000Z"


@pytest.fixture
def runtime(tmp_path):
    data = load_json_object(ROOT / "configs/runtime.json")
    data["storage"]["runtime_root"] = str(tmp_path)
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(data))
    return RuntimeConfig.from_dict(data, root=ROOT, path=path)


@pytest.mark.parametrize("name,count", [("memory", 80), ("memory-cpu", 64), ("throughput", 4), ("throughput-cpu", 4), ("rag", 4)])
def test_authored_sweeps_expand_to_complete_unique_grids(runtime, name, count):
    plan = load_json_object(ROOT / f"configs/launch/{name}.json")
    original = deepcopy(plan)
    runs = launcher.expand_runs(plan, runtime, "run", STAMP)
    assert len(runs) == count
    assert len({run["run_id"] for run in runs}) == count
    assert original == plan
    if name.endswith("cpu"):
        assert all(run["inferscale"]["kv"]["store_backend"] == "cpu" for run in runs)
    if name == "memory":
        assert {run["top_k"] for run in runs} == {5, 10, 20, 50}
        assert {run["context_window"] for run in runs} == {0, 5, 20, 50}
        assert all(run["skip_judge"] for run in runs)


def test_precompute_covers_each_window_once_per_model(runtime):
    plan = load_json_object(ROOT / "configs/launch/memory.json")
    runs = launcher.expand_runs(plan, runtime, "precompute-kv", STAMP)
    assert len(runs) == 16
    assert len({(run["model"], run["context_window"]) for run in runs}) == 16


def test_manifest_preserves_concrete_inputs_for_judge(runtime, tmp_path, monkeypatch, capsys):
    plan = load_json_object(ROOT / "configs/launch/memory.json")
    plan["configs"] = plan["configs"][:1]
    plan["stamp"] = STAMP
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    manifest, path = launcher.materialize_plan(source, runtime, "run")
    original = Path(manifest["run_configs"][0]).read_bytes()
    monkeypatch.setattr(launcher, "run_job", lambda *args: pytest.fail("Dry run started a subprocess"))
    assert launcher.execute_manifest(manifest, path, runtime, "judge", dry_run=True) == 0
    replay, replay_path = launcher.materialize_plan(path, runtime, "judge")
    assert replay == manifest
    assert replay_path == path
    assert Path(manifest["run_configs"][0]).read_bytes() == original
    judge_path = path.parent / "judge-inputs" / Path(manifest["run_configs"][0]).name
    judge_data = load_json_object(judge_path)
    run_data = json.loads(original)
    assert judge_data["run_id"] == run_data["run_id"]
    assert run_data.pop("skip_judge") is True
    assert judge_data.pop("skip_judge") is False
    assert judge_data == run_data
    output = capsys.readouterr().out
    jobs = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    assert len(jobs) == len(manifest["run_configs"])
    assert all(job["stage"] == "judge" for job in jobs)


def test_duplicate_run_ids_and_invalid_stages_fail_before_launch(runtime):
    path = "configs/memory/accuracy-latency/llama.json"
    plan = {"family": "memory", "configs": [path, path]}
    with pytest.raises(ConfigError, match="duplicate run IDs"):
        launcher.expand_runs(plan, runtime, "extract", STAMP)
    with pytest.raises(ConfigError, match="not supported"):
        launcher.expand_runs(plan, runtime, "prepare", STAMP)


def test_extraction_uses_json_endpoint_and_rejects_port_mismatch(runtime):
    data = load_json_object(ROOT / "configs/memory/accuracy-latency/llama.json")
    command, config = launcher.extraction_command(data, runtime)
    assert command[:3] == ["vllm", "serve", runtime.resolve_model("llama")]
    assert command[command.index("--max-model-len") + 1] == "16384"
    assert config.memory_llm_base_url == "http://localhost:8000/v1"
    data["mem0"]["llm_base_url"] = "http://localhost:8123/v1"
    with pytest.raises(ConfigError, match="extraction.port"):
        launcher.extraction_command(data, runtime)


def test_extraction_server_is_stopped_when_preembed_fails(runtime, tmp_path, monkeypatch):
    data = load_json_object(ROOT / "configs/memory/accuracy-latency/llama.json")
    calls = iter([False, True])
    monkeypatch.setattr(launcher, "_server_alive", lambda *_args: next(calls))
    process = SimpleNamespace(pid=123)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: process)
    stopped = []
    monkeypatch.setattr(launcher, "stop_server", stopped.append)
    def fail(*args):
        raise RuntimeError("preembed failed")
    monkeypatch.setattr(launcher, "run_job", fail)
    with pytest.raises(RuntimeError, match="preembed failed"):
        launcher.run_extraction(data, runtime, {"family": "memory", "stage": "preembed"}, tmp_path / "extract.log")
    assert stopped == [process]


def test_stop_server_terminates_process_group_and_waits(monkeypatch):
    class Process:
        pid = 99
        def poll(self):
            return None
        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("vllm", timeout)
    signals = []
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    launcher.stop_server(Process())
    assert signals == [(99, launcher.signal.SIGTERM), (99, launcher.signal.SIGKILL)]


def test_direct_benchmark_config_selects_one_run_without_implicit_sweep(runtime):
    source = ROOT / "configs/memory/accuracy-latency/llama.json"
    manifest, _ = launcher.materialize_plan(source, runtime, "run")
    assert len(manifest["run_configs"]) == 1
    data = load_json_object(manifest["run_configs"][0])
    assert data["top_k"] == 50
    assert data["context_window"] == 0
    assert data["skip_judge"] is False
    assert data["model"] == runtime.resolve_model("llama")


def test_changed_judge_runtime_preserves_answer_model_data_and_results(runtime, tmp_path):
    from benchmarks.memory.config import MemoryRunConfig
    source = ROOT / "configs/memory/accuracy-latency/llama.json"
    manifest, path = launcher.materialize_plan(source, runtime, "run")
    data = load_json_object(manifest["run_configs"][0])
    before = MemoryRunConfig.from_dict(data, runtime=runtime)
    changed = load_json_object(runtime.path)
    changed["storage"]["runtime_root"] = str(tmp_path / "another-root")
    changed["models"]["llama"] = "another/model"
    changed["judge_server"]["base_url"] = "http://127.0.0.1:9999/v1"
    new_runtime = RuntimeConfig.from_dict(changed, root=ROOT, path=tmp_path / "new-runtime.json")
    launcher.execute_manifest(manifest, path, new_runtime, "judge", dry_run=True)
    judge_input = load_json_object(path.parent / "judge-inputs" / Path(manifest["run_configs"][0]).name)
    after = MemoryRunConfig.from_dict(judge_input, runtime=new_runtime)
    assert (after.model, after.dataset_path, after.results_dir, after.run_id) == (before.model, before.dataset_path, before.results_dir, before.run_id)
    assert after.judge_base_url == "http://127.0.0.1:9999/v1"


@pytest.mark.parametrize("change", [
    {"family": []}, {"name": 123}, {"sweep": []}, {"configs": []},
    {"configs": None}, {"stage": "unknown"}, {"stamp": {}}, {"dry_run": "true"},
    {"dry_run": 1}, {"manifest": "saved.json"},
])
def test_malformed_launch_values_have_configuration_errors(runtime, change):
    plan = {"family": "memory", "configs": ["configs/memory/accuracy-latency/llama.json"]}
    plan.update(change)
    with pytest.raises(ConfigError):
        launcher.expand_runs(plan, runtime, "run", STAMP)


@pytest.mark.parametrize("family,stage", [("memory", "estimate"), ("rag", "check-catalogs"), ("throughput", "judge")])
def test_saved_manifest_rejects_invalid_stage_before_any_subprocess(runtime, tmp_path, monkeypatch, family, stage):
    monkeypatch.setattr(launcher, "run_job", lambda *args: pytest.fail("Invalid stage started a subprocess"))
    manifest = {"family": family, "run_configs": [str(tmp_path / "must-not-be-read.json")]}
    with pytest.raises(ConfigError, match="not supported"):
        launcher.execute_manifest(manifest, tmp_path / "manifest.json", runtime, stage, dry_run=True)


def test_json_launch_controls_stage_dry_run_and_runtime(runtime, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher.os, "environ", launcher.os.environ.copy())
    plan = {
        "family": "memory", "configs": ["configs/memory/accuracy-latency/llama.json"],
        "runtime_config": str(runtime.path), "stage": "precompute-kv", "dry_run": True,
        "stamp": STAMP,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(launcher, "run_job", lambda *_: pytest.fail("Dry run started a subprocess"))

    assert launcher.launch(path) == 0

    jobs = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(jobs) == 4
    assert all(job["stage"] == "precompute-kv" for job in jobs)
    assert all(str(tmp_path) in job["runtime_config"] for job in jobs)
    assert len({load_json_object(job["config"])["context_window"] for job in jobs}) == 4


def test_json_judge_plan_replays_saved_runs_with_selected_runtime(runtime, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher.os, "environ", launcher.os.environ.copy())
    manifest, manifest_path = launcher.materialize_plan(
        ROOT / "configs/memory/accuracy-latency/llama.json", runtime, "run",
    )
    original_path = Path(manifest["run_configs"][0])
    original = original_path.read_bytes()
    changed = load_json_object(runtime.path)
    changed["storage"]["runtime_root"] = str(tmp_path / "new-results")
    changed["models"]["llama"] = "another/model"
    changed["judge_server"]["base_url"] = "http://127.0.0.1:9999/v1"
    new_runtime = tmp_path / "judge-runtime.json"
    new_runtime.write_text(json.dumps(changed))
    plan_path = tmp_path / "judge-plan.json"
    plan_path.write_text(json.dumps({
        "family": "memory", "manifest": str(manifest_path), "stage": "judge", "dry_run": True,
        "runtime_config": str(new_runtime),
    }))
    monkeypatch.setattr(launcher, "run_job", lambda *_: pytest.fail("Dry run started a subprocess"))

    assert launcher.launch(plan_path) == 0

    jobs = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert len(jobs) == 1
    assert jobs[0]["runtime_config"] == str(new_runtime)
    assert jobs[0]["stage"] == "judge"
    replay = load_json_object(jobs[0]["config"])
    original_data = json.loads(original)
    for name in ("model", "dataset_path", "run_id", "results_dir"):
        assert replay[name] == original_data[name]
    assert replay["skip_judge"] is False
    assert original_path.read_bytes() == original


def test_json_job_crosses_process_boundary_and_streams_logs(runtime, tmp_path, capsys):
    dataset = tmp_path / "empty-locomo.json"
    dataset.write_text("[]")
    data = load_json_object(ROOT / "configs/memory/accuracy-latency/llama.json")
    data.update(dataset_path=str(dataset), run_id="catalog-check")
    config_path = tmp_path / "memory.json"
    config_path.write_text(json.dumps(data))
    log_file = tmp_path / "logs" / "check.log"

    assert launcher.run_job({
        "family": "memory", "config": str(config_path), "runtime_config": str(runtime.path),
        "stage": "check-catalogs",
    }, log_file) == 0

    assert "fact catalogs complete" in log_file.read_text()
    assert capsys.readouterr().out == log_file.read_text()
    assert not (runtime.layout.results_root / "catalog-check").exists()


def test_failed_json_job_returns_nonzero_and_retains_error_log(runtime, tmp_path):
    log_file = tmp_path / "logs" / "invalid.log"
    assert launcher.run_job({
        "family": "memory", "config": str(ROOT / "configs/memory/accuracy-latency/llama.json"),
        "runtime_config": str(runtime.path), "stage": "unknown",
    }, log_file) != 0
    assert "Unsupported memory stage" in log_file.read_text()


@pytest.mark.parametrize("family", ["memory", "rag"])
@pytest.mark.parametrize("stage,change", [
    ("run", {"rejudge": True}),
    ("preembed", {"rejudge": True}),
    ("judge", {"run_id": None}),
])
def test_later_invalid_input_is_rejected_before_any_process(runtime, tmp_path, monkeypatch, family, stage, change):
    config_path = (
        ROOT / "configs/memory/accuracy-latency/llama.json" if family == "memory"
        else ROOT / "configs/rag/multihoprag.json"
    )
    manifest, path = launcher.materialize_plan(config_path, runtime, "run")
    invalid = load_json_object(manifest["run_configs"][0])
    invalid.update(change)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid))
    manifest["run_configs"].append(str(invalid_path))
    monkeypatch.setattr(launcher, "run_job", lambda *_: pytest.fail("Validation started a process"))

    with pytest.raises(ConfigError):
        launcher.execute_manifest(manifest, path, runtime, stage, dry_run=False)

    assert not (path.parent / "judge-inputs").exists()


def test_runtime_selection_comes_from_json_not_legacy_environment(monkeypatch):
    from benchmarks.common.config import load_runtime_config
    monkeypatch.setenv("INFERSCALE_RUNTIME_CONFIG", "/missing/old-runtime.json")
    assert load_runtime_config().path == ROOT / "configs/runtime.json"
