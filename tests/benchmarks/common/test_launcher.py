import json
import os
import re
import signal
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig, load_json_object
from benchmarks.common.launcher import execute, extraction, plans, processes
from benchmarks.common.launcher.manifest import materialize_plan

ROOT = Path(__file__).resolve().parents[3]
MEMORY_CONFIG = ROOT / "configs/memory/accuracy-latency/llama.json"
STAMP = "20260908T120000Z"
LAUNCH_PLAN_FROM_ARGV = """
import sys
from benchmarks.common.launcher import launch

raise SystemExit(launch(sys.argv[1]))
"""
LOG_LINE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \| INFO  \| (?P<message>.+)")


def announced_jobs(log_messages):
    return [json.loads(message) for message in log_messages if message.startswith("{")]


@pytest.mark.parametrize(
    "name,count",
    [
        ("memory", 168),
        ("memory-cpu", 4),
        ("memory-lmcache", 28),
        ("memory-jasper-prompt-injection", 28),
        ("throughput", 4),
        ("throughput-cpu", 4),
        ("throughput-lmcache", 4),
        ("rag", 4),
        ("rag-lmcache", 2),
    ],
)
def test_authored_sweeps_expand_to_complete_unique_grids(runtime, name, count):
    plan = load_json_object(ROOT / f"configs/launch/{name}.json")
    original = deepcopy(plan)
    runs = plans.expand_runs(plan, runtime, "run", STAMP)
    assert len(runs) == count
    assert len({run["run_id"] for run in runs}) == count
    assert original == plan
    if name.endswith("cpu"):
        assert all(run["inferscale"]["kv"]["store_backend"] == "cpu" for run in runs)
    if name.endswith("lmcache"):
        assert all(run["lmcache"]["max_local_cpu_size_gb"] > 0 for run in runs)
    if name == "memory-lmcache":
        assert all(run["answer_backend"] == "lmcache-blend" for run in runs)
        assert {run["top_k"] for run in runs} == {5, 10, 20, 50, 100, 200, 400}
        assert all(run["context_window"] == 0 for run in runs)
        assert all("-lmcache-blend-mem0-qdrant10-" in run["run_id"] for run in runs)
    if name == "memory-jasper-prompt-injection":
        default_plan = load_json_object(ROOT / "configs/launch/memory.json")
        assert runs == [
            run
            for run in plans.expand_runs(default_plan, runtime, "run", STAMP)
            if (run["answer_backend"], run["vector_backend"]) == ("prompt-injection", "jasper")
        ]
    if name == "throughput-lmcache":
        assert all(run["conditions"] == ["lmcache_blend"] for run in runs)
    if name == "rag-lmcache":
        assert all(run["answer_backend"] == "lmcache-blend" for run in runs)
    if name == "memory":
        assert {run["top_k"] for run in runs} == {5, 10, 20, 50, 100, 200, 400}
        assert {run["context_window"] for run in runs} == {0, 5, 20, 50}
        assert all(
            "-prompt-injection-mem0-" in run["run_id"]
            for run in runs
            if run["answer_backend"] == "prompt-injection"
        )
        assert all(run["skip_judge"] for run in runs)


def test_precompute_covers_each_window_once_per_model(runtime):
    plan = load_json_object(ROOT / "configs/launch/memory.json")
    runs = plans.expand_runs(plan, runtime, "precompute-kv", STAMP)
    assert len(runs) == 16
    assert len({(run["model"], run["context_window"]) for run in runs}) == 16


def test_extract_run_uses_the_extraction_workers_and_the_qdrant_backend(runtime, tmp_path):
    data = load_json_object(MEMORY_CONFIG)
    data["extraction"] = {**data["extraction"], "workers": 7}
    source = tmp_path / "memory.json"
    source.write_text(json.dumps(data))
    plan = {"family": "memory", "configs": [str(source)]}

    [extract_run] = plans.expand_runs(plan, runtime, "extract", STAMP)
    [preembed_run] = plans.expand_runs(plan, runtime, "preembed", STAMP)

    assert (extract_run["preembed_workers"], extract_run["vector_backend"]) == (7, "qdrant")
    assert (preembed_run["preembed_workers"], preembed_run["vector_backend"]) == (4, "jasper")
    assert extract_run["run_id"] == f"extract-llama-{STAMP}"


def test_manifest_preserves_concrete_inputs_for_judge(runtime, tmp_path, monkeypatch, launcher_log):
    plan = load_json_object(ROOT / "configs/launch/memory.json")
    plan["configs"] = plan["configs"][:1]
    plan["stamp"] = STAMP
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    manifest, path = materialize_plan(source, runtime, "run")
    original = Path(manifest["run_configs"][0]).read_bytes()
    monkeypatch.setattr(
        execute, "run_job", lambda *args: pytest.fail("Dry run started a subprocess")
    )
    assert execute.execute_manifest(manifest, path, runtime, "judge", dry_run=True) == 0
    replay, replay_path = materialize_plan(path, runtime, "judge")
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
    jobs = announced_jobs(launcher_log)
    assert len(jobs) == len(manifest["run_configs"])
    assert all(job["stage"] == "judge" for job in jobs)


def test_duplicate_run_ids_and_invalid_stages_fail_before_launch(runtime):
    path = "configs/memory/accuracy-latency/llama.json"
    plan = {"family": "memory", "configs": [path, path]}
    with pytest.raises(ConfigError, match="duplicate run IDs"):
        plans.expand_runs(plan, runtime, "extract", STAMP)
    with pytest.raises(ConfigError, match="not supported"):
        plans.expand_runs(plan, runtime, "prepare", STAMP)


def test_extraction_uses_json_endpoint_and_rejects_port_mismatch(runtime):
    data = load_json_object(ROOT / "configs/memory/accuracy-latency/llama.json")
    command, config = extraction.extraction_command(data, runtime)
    assert command[:3] == ["vllm", "serve", runtime.resolve_model("llama")]
    assert command[command.index("--max-model-len") + 1] == "16384"
    assert config.memory_llm_base_url == "http://localhost:8000/v1"
    data["mem0"]["llm_base_url"] = "http://localhost:8123/v1"
    with pytest.raises(ConfigError, match="extraction.port"):
        extraction.extraction_command(data, runtime)


def memory_config_overriding_the_extraction_server_port(tmp_path):
    data = load_json_object(MEMORY_CONFIG)
    data["extraction"] = {**data.get("extraction", {}), "extra_vllm_args": ["--port=8123"]}
    source = tmp_path / "memory.json"
    source.write_text(json.dumps(data))
    return source


def test_extract_plan_with_unusable_server_settings_writes_no_launch_folder(runtime, tmp_path):
    source = memory_config_overriding_the_extraction_server_port(tmp_path)

    with pytest.raises(ConfigError, match="extra_vllm_args cannot override"):
        materialize_plan(source, runtime, "extract")

    assert not (runtime.layout.results_root / "launches").exists()


def test_saved_runs_with_unusable_server_settings_are_rejected_before_extraction_starts(
    runtime, tmp_path, monkeypatch
):
    source = memory_config_overriding_the_extraction_server_port(tmp_path)
    manifest, path = materialize_plan(source, runtime, "run")
    monkeypatch.setattr(
        execute, "run_extraction", lambda *_: pytest.fail("Extraction started a server")
    )

    with pytest.raises(ConfigError, match="extra_vllm_args cannot override"):
        execute.execute_manifest(manifest, path, runtime, "extract", dry_run=False)


def test_judge_plan_over_a_config_without_run_id_is_rejected(runtime):
    plan = {"family": "memory", "configs": [str(MEMORY_CONFIG)]}

    with pytest.raises(ConfigError, match="existing run_id"):
        plans.expand_runs(plan, runtime, "judge", STAMP)


def test_extraction_server_is_stopped_when_preembed_fails(runtime, tmp_path, monkeypatch):
    data = load_json_object(ROOT / "configs/memory/accuracy-latency/llama.json")
    calls = iter([False, True])
    monkeypatch.setattr(extraction, "server_alive", lambda *_args: next(calls))
    process = SimpleNamespace(pid=123)
    monkeypatch.setattr(extraction.subprocess, "Popen", lambda *args, **kwargs: process)
    stopped = []
    monkeypatch.setattr(extraction, "stop_server", stopped.append)

    def fail(*args):
        raise RuntimeError("preembed failed")

    monkeypatch.setattr(extraction, "run_job", fail)
    with pytest.raises(RuntimeError, match="preembed failed"):
        extraction.run_extraction(
            data, runtime, {"family": "memory", "stage": "preembed"}, tmp_path / "extract.log"
        )
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
    monkeypatch.setattr(processes.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    processes.stop_server(Process())
    assert signals == [(99, signal.SIGTERM), (99, signal.SIGKILL)]


def test_saved_manifest_with_a_missing_run_config_is_rejected(runtime):
    manifest, path = materialize_plan(MEMORY_CONFIG, runtime, "run")
    Path(manifest["run_configs"][0]).unlink()

    with pytest.raises(ConfigError, match="Saved run config not found"):
        materialize_plan(path, runtime, "judge")


@pytest.mark.parametrize("family,referenced", [("rag", "manifest"), ("memory", "launch plan")])
def test_manifest_reference_must_name_a_saved_manifest_of_the_same_family(
    runtime, tmp_path, family, referenced
):
    _, manifest_path = materialize_plan(MEMORY_CONFIG, runtime, "run")
    referenced_paths = {
        "manifest": manifest_path,
        "launch plan": ROOT / "configs/launch/memory.json",
    }
    plan = {"family": family, "manifest": str(referenced_paths[referenced]), "stage": "judge"}
    plan_path = tmp_path / "judge-plan.json"
    plan_path.write_text(json.dumps(plan))

    with pytest.raises(ConfigError, match="must reference a saved manifest for the same"):
        materialize_plan(plan_path, runtime, "judge")


def test_direct_benchmark_config_selects_one_run_without_implicit_sweep(runtime):
    source = ROOT / "configs/memory/accuracy-latency/llama.json"
    manifest, _ = materialize_plan(source, runtime, "run")
    assert len(manifest["run_configs"]) == 1
    data = load_json_object(manifest["run_configs"][0])
    assert data["top_k"] == 50
    assert data["context_window"] == 0
    assert data["skip_judge"] is False
    assert data["model"] == runtime.resolve_model("llama")


def test_changed_judge_runtime_preserves_answer_model_data_and_results(runtime, tmp_path):
    from benchmarks.memory.config import MemoryRunConfig

    source = ROOT / "configs/memory/accuracy-latency/llama.json"
    manifest, path = materialize_plan(source, runtime, "run")
    data = load_json_object(manifest["run_configs"][0])
    before = MemoryRunConfig.from_dict(data, runtime=runtime)
    changed = load_json_object(runtime.path)
    changed["storage"]["runtime_root"] = str(tmp_path / "another-root")
    changed["models"]["llama"] = "another/model"
    changed["judge_server"]["base_url"] = "http://127.0.0.1:9999/v1"
    new_runtime = RuntimeConfig.from_dict(changed, root=ROOT, path=tmp_path / "new-runtime.json")
    execute.execute_manifest(manifest, path, new_runtime, "judge", dry_run=True)
    judge_input = load_json_object(
        path.parent / "judge-inputs" / Path(manifest["run_configs"][0]).name
    )
    after = MemoryRunConfig.from_dict(judge_input, runtime=new_runtime)
    assert (after.model, after.dataset_path, after.results_dir, after.run_id) == (
        before.model,
        before.dataset_path,
        before.results_dir,
        before.run_id,
    )
    assert after.judge_base_url == "http://127.0.0.1:9999/v1"


@pytest.mark.parametrize(
    "change",
    [
        {"family": []},
        {"name": 123},
        {"sweep": []},
        {"configs": []},
        {"configs": None},
        {"stage": "unknown"},
        {"stamp": {}},
        {"dry_run": "true"},
        {"dry_run": 1},
        {"manifest": "saved.json"},
    ],
)
def test_malformed_launch_values_have_configuration_errors(runtime, change):
    plan = {"family": "memory", "configs": ["configs/memory/accuracy-latency/llama.json"]}
    plan.update(change)
    with pytest.raises(ConfigError):
        plans.expand_runs(plan, runtime, "run", STAMP)


@pytest.mark.parametrize(
    "family,stage", [("memory", "estimate"), ("rag", "check-catalogs"), ("throughput", "judge")]
)
def test_saved_manifest_rejects_invalid_stage_before_any_subprocess(
    runtime, tmp_path, monkeypatch, family, stage
):
    monkeypatch.setattr(
        execute, "run_job", lambda *args: pytest.fail("Invalid stage started a subprocess")
    )
    manifest = {"family": family, "run_configs": [str(tmp_path / "must-not-be-read.json")]}
    with pytest.raises(ConfigError, match="not supported"):
        execute.execute_manifest(manifest, tmp_path / "manifest.json", runtime, stage, dry_run=True)


def test_json_launch_controls_stage_dry_run_and_runtime(
    runtime, tmp_path, monkeypatch, launcher_log
):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    plan = {
        "family": "memory",
        "configs": ["configs/memory/accuracy-latency/llama.json"],
        "runtime_config": str(runtime.path),
        "stage": "precompute-kv",
        "dry_run": True,
        "stamp": STAMP,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(execute, "run_job", lambda *_: pytest.fail("Dry run started a subprocess"))

    assert execute.launch(path) == 0

    jobs = announced_jobs(launcher_log)
    assert len(jobs) == 4
    assert all(job["stage"] == "precompute-kv" for job in jobs)
    launch_folder = Path(jobs[0]["config"]).parents[1]
    assert {job["runtime_config"] for job in jobs} == {str(launch_folder / "runtime.json")}
    assert load_json_object(launch_folder / "runtime.json") == load_json_object(runtime.path)
    assert len({load_json_object(job["config"])["context_window"] for job in jobs}) == 4


def test_json_judge_plan_replays_saved_runs_with_selected_runtime(
    runtime, tmp_path, monkeypatch, launcher_log
):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    manifest, manifest_path = materialize_plan(
        ROOT / "configs/memory/accuracy-latency/llama.json",
        runtime,
        "run",
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
    plan_path.write_text(
        json.dumps(
            {
                "family": "memory",
                "manifest": str(manifest_path),
                "stage": "judge",
                "dry_run": True,
                "runtime_config": str(new_runtime),
            }
        )
    )
    monkeypatch.setattr(execute, "run_job", lambda *_: pytest.fail("Dry run started a subprocess"))

    assert execute.launch(plan_path) == 0

    jobs = announced_jobs(launcher_log)
    assert len(jobs) == 1
    assert jobs[0]["runtime_config"] == str(new_runtime)
    assert jobs[0]["stage"] == "judge"
    replay = load_json_object(jobs[0]["config"])
    original_data = json.loads(original)
    for name in ("model", "dataset_path", "run_id", "results_dir"):
        assert replay[name] == original_data[name]
    assert replay["skip_judge"] is False
    assert original_path.read_bytes() == original


def test_json_job_crosses_process_boundary_and_streams_logs(runtime, tmp_path, launcher_log):
    dataset = tmp_path / "empty-locomo.json"
    dataset.write_text("[]")
    data = load_json_object(ROOT / "configs/memory/accuracy-latency/llama.json")
    data.update(dataset_path=str(dataset), run_id="catalog-check")
    config_path = tmp_path / "memory.json"
    config_path.write_text(json.dumps(data))
    log_file = tmp_path / "logs" / "check.log"

    assert (
        processes.run_job(
            {
                "family": "memory",
                "config": str(config_path),
                "runtime_config": str(runtime.path),
                "stage": "check-catalogs",
            },
            log_file,
        )
        == 0
    )

    assert "fact catalogs complete" in log_file.read_text()
    assert "".join(launcher_log) == log_file.read_text()
    assert not (runtime.layout.results_root / "catalog-check").exists()


def test_failed_json_job_returns_nonzero_and_retains_error_log(runtime, tmp_path):
    log_file = tmp_path / "logs" / "invalid.log"
    assert (
        processes.run_job(
            {
                "family": "memory",
                "config": str(ROOT / "configs/memory/accuracy-latency/llama.json"),
                "runtime_config": str(runtime.path),
                "stage": "unknown",
            },
            log_file,
        )
        != 0
    )
    assert "Unsupported memory stage" in log_file.read_text()


@pytest.mark.parametrize("family", ["memory", "rag"])
@pytest.mark.parametrize(
    "stage,change",
    [
        ("run", {"rejudge": True}),
        ("preembed", {"rejudge": True}),
        ("judge", {"run_id": None}),
    ],
)
def test_later_invalid_input_is_rejected_before_any_process(
    runtime, tmp_path, monkeypatch, family, stage, change
):
    config_path = (
        ROOT / "configs/memory/accuracy-latency/llama.json"
        if family == "memory"
        else ROOT / "configs/rag/multihoprag.json"
    )
    manifest, path = materialize_plan(config_path, runtime, "run")
    invalid = load_json_object(manifest["run_configs"][0])
    invalid.update(change)
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid))
    manifest["run_configs"].append(str(invalid_path))
    monkeypatch.setattr(execute, "run_job", lambda *_: pytest.fail("Validation started a process"))

    with pytest.raises(ConfigError):
        execute.execute_manifest(manifest, path, runtime, stage, dry_run=False)

    assert not (path.parent / "judge-inputs").exists()


def record_started_jobs(monkeypatch, *, fails=lambda job: False):
    started = []

    def run_job(job, log_file):
        started.append((job["config"], job["stage"], log_file.name))
        return 1 if fails(job) else 0

    monkeypatch.setattr(execute, "run_job", run_job)
    return started


def test_memory_run_checks_fact_catalogs_before_the_run_job(runtime, monkeypatch):
    manifest, path = materialize_plan(MEMORY_CONFIG, runtime, "run")
    run_config_path = manifest["run_configs"][0]
    run_id = load_json_object(run_config_path)["run_id"]
    started = record_started_jobs(monkeypatch)

    assert execute.execute_manifest(manifest, path, runtime, "run", dry_run=False) == 0

    assert started == [
        (run_config_path, "check-catalogs", f"{run_id}.run.catalogs.log"),
        (run_config_path, "run", f"{run_id}.run.log"),
    ]


@pytest.mark.parametrize(
    "failing_stage,stages_of_failed_run",
    [("check-catalogs", ["check-catalogs"]), ("run", ["check-catalogs", "run"])],
)
def test_failed_run_fails_the_launch_without_stopping_later_runs(
    runtime, tmp_path, monkeypatch, launcher_log, failing_stage, stages_of_failed_run
):
    plan = load_json_object(ROOT / "configs/launch/memory.json")
    plan["configs"] = plan["configs"][:1]
    plan["stamp"] = STAMP
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(plan))
    manifest, path = materialize_plan(source, runtime, "run")
    failed, later = manifest["run_configs"][:2]
    manifest["run_configs"] = [failed, later]
    started = record_started_jobs(
        monkeypatch,
        fails=lambda job: job["config"] == failed and job["stage"] == failing_stage,
    )

    assert execute.execute_manifest(manifest, path, runtime, "run", dry_run=False) == 1

    assert [(config, stage) for config, stage, _ in started] == [
        *((failed, stage) for stage in stages_of_failed_run),
        (later, "check-catalogs"),
        (later, "run"),
    ]
    assert f"Failed runs: {load_json_object(failed)['run_id']}" in launcher_log
    assert launcher_log[-1] == f"Launch manifest: {path}"


def test_extract_runs_the_preembed_job_behind_the_extraction_server(
    runtime, monkeypatch, launcher_log
):
    manifest, path = materialize_plan(MEMORY_CONFIG, runtime, "extract")
    run_config_path = manifest["run_configs"][0]
    run_id = load_json_object(run_config_path)["run_id"]
    monkeypatch.setattr(
        execute, "run_job", lambda *_: pytest.fail("Extraction bypassed its server")
    )
    served = []

    def run_extraction(data, runtime, job, log_file):
        served.append((data["run_id"], job["stage"], log_file.name))
        return 0

    monkeypatch.setattr(execute, "run_extraction", run_extraction)

    assert execute.execute_manifest(manifest, path, runtime, "extract", dry_run=False) == 0

    assert served == [(run_id, "preembed", f"{run_id}.extract.log")]
    progress, server_command, job, manifest_location = launcher_log
    assert progress == f"[1/1] {run_id}"
    assert server_command.startswith(f"vllm serve {runtime.resolve_model('llama')} ")
    assert json.loads(job) == {
        "family": "memory",
        "config": run_config_path,
        "runtime_config": str(runtime.path),
        "stage": "preembed",
    }
    assert manifest_location == f"Launch manifest: {path}"


def test_launch_logs_to_stderr_in_the_benchmark_log_format(runtime, tmp_path):
    plan = {
        "family": "memory",
        "configs": [str(MEMORY_CONFIG)],
        "runtime_config": str(runtime.path),
        "stage": "precompute-kv",
        "dry_run": True,
        "stamp": STAMP,
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    process = subprocess.run(
        [sys.executable, "-c", LAUNCH_PLAN_FROM_ARGV, str(plan_path)],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "src")])},
        text=True,
        capture_output=True,
    )

    assert process.returncode == 0, process.stderr
    assert process.stdout == ""
    log_lines = filter(None, map(LOG_LINE.fullmatch, process.stderr.splitlines()))
    messages = [log_line["message"] for log_line in log_lines]
    assert len(announced_jobs(messages)) == 4
    assert messages[0].startswith("[1/4] precompute-")
    assert messages[-1].startswith("Launch manifest: ")


def test_runtime_selection_comes_from_json_not_legacy_environment(monkeypatch):
    from benchmarks.common.config import load_runtime_config

    monkeypatch.setenv("INFERSCALE_RUNTIME_CONFIG", "/missing/old-runtime.json")
    assert load_runtime_config().path == ROOT / "configs/runtime.json"
