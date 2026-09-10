"""Expand JSON benchmark plans and preserve the inputs for later judge passes."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from benchmarks.common.cache_identity import atomic_write_json
from benchmarks.common.config import (
    ConfigError, RuntimeConfig, expand_path, load_json_object, load_runtime_config,
    optional, reject_unknown_keys, require, slug, stamp_now,
)

STAGES = ("run", "extract", "check-catalogs", "preembed", "precompute-kv", "prepare", "estimate", "judge")
FAMILY_STAGES = {
    "memory": {"run", "judge", "extract", "check-catalogs", "preembed", "precompute-kv"},
    "throughput": {"run"},
    "rag": {"run", "judge", "prepare", "estimate", "preembed", "precompute-kv"},
}


def validate_stage(family: str, stage: str) -> None:
    if stage not in FAMILY_STAGES.get(family, set()):
        raise ConfigError(f"Stage {stage!r} is not supported by {family} launch plans.")


def validate_plan(plan: dict[str, Any]) -> None:
    reject_unknown_keys(plan, ("family", "name", "configs", "sweep", "stamp", "run_configs", "runtime_config", "stage", "dry_run", "manifest"), "launch")
    family = require(plan, "family", str, "launch")
    if family not in FAMILY_STAGES:
        raise ConfigError(f"Unknown launch family {family!r}.")
    for name in ("name", "sweep", "stamp", "runtime_config", "stage", "manifest"):
        value = optional(plan, name, str, None, "launch")
        if value is not None and not value.strip():
            raise ConfigError(f"launch.{name} must not be empty.")
    if plan.get("stage") is not None and plan["stage"] not in STAGES:
        raise ConfigError(f"Unknown launch stage {plan['stage']!r}.")
    optional(plan, "dry_run", bool, False, "launch")
    if "manifest" in plan:
        require(plan, "manifest", str, "launch")
        if any(key in plan for key in ("configs", "sweep", "run_configs")):
            raise ConfigError("A launch manifest reference cannot also specify configs, run_configs or a sweep.")
        return
    if "run_configs" in plan:
        if "configs" in plan or "sweep" in plan:
            raise ConfigError("Saved launch manifests cannot also specify configs or a sweep.")
        require(plan, "runtime_config", str, "launch")
        key = "run_configs"
    else:
        key = "configs"
    paths = require(plan, key, list, "launch")
    if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ConfigError(f"launch.{key} must be a non-empty list of JSON config paths.")


def _family(data: dict[str, Any]) -> str:
    benchmark = data.get("benchmark")
    if not isinstance(benchmark, str):
        return ""
    return {"memory-accuracy-latency": "memory", "memory-throughput": "throughput", "rag": "rag"}.get(benchmark, "")


def load_launch_plan(path: Path) -> dict[str, Any]:
    data = load_json_object(path)
    family = _family(data)
    if family:
        return {"family": family, "name": f"{family}-{slug(path.stem)}", "configs": [str(path.resolve())]}
    return data


def validate_run(data: dict[str, Any], runtime: RuntimeConfig, family: str, stage: str) -> None:
    """Check the exact stage input before any benchmark processes are started."""
    if _family(data) != family:
        raise ConfigError(f"Saved input does not match launch family {family}.")
    if not data.get("run_id"):
        raise ConfigError("Materialized benchmark inputs require an explicit run_id.")
    actual_stage = "preembed" if stage in {"extract", "prepare"} else stage
    if stage == "judge":
        data = {**data, "skip_judge": False}
    if family == "memory":
        from benchmarks.memory.config import MemoryRunConfig, validate_memory_stage
        config = MemoryRunConfig.from_dict(data, runtime=runtime)
        validate_memory_stage(config, actual_stage, has_run_id=True)
    elif family == "rag":
        from benchmarks.rag.config import RagBenchConfig, validate_rag_stage
        config = RagBenchConfig.from_dict(data, runtime=runtime)
        validate_rag_stage(config, actual_stage, has_run_id=True)
    else:
        from benchmarks.memory.throughput.config import ThroughputConfig
        ThroughputConfig.from_dict(data, runtime=runtime)


def expand_runs(plan: dict[str, Any], runtime: RuntimeConfig, stage: str, stamp: str) -> list[dict[str, Any]]:
    """Materialize only sweep axes authored in JSON, without changing the source."""
    validate_plan(plan)
    family = plan["family"]
    validate_stage(family, stage)
    paths = plan.get("configs")
    if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
        raise ConfigError("launch.configs must be a non-empty list of JSON config paths.")
    runs: list[dict[str, Any]] = []
    for path in paths:
        data = load_json_object(expand_path(path, root=runtime.root))
        if _family(data) != family:
            raise ConfigError(f"{path} does not contain a {family} benchmark config.")
        if family == "memory":
            from benchmarks.memory.config import MemoryRunConfig
            config = MemoryRunConfig.from_dict(data, runtime=runtime)
            if stage == "run" and plan.get("sweep"):
                for cell in config.sweep_cells(plan["sweep"]):
                    item = cell.materialize(data)
                    item["run_id"] = cell.run_id(config.label, config.max_samples, stamp)
                    item["skip_judge"] = True
                    MemoryRunConfig.from_dict(item, runtime=runtime)
                    runs.append(item)
            elif stage == "precompute-kv":
                for window in config.context_windows():
                    item = deepcopy(data)
                    item["context_window"] = window
                    item["run_id"] = f"precompute-{config.label}-s{window}-{stamp}"
                    MemoryRunConfig.from_dict(item, runtime=runtime)
                    runs.append(item)
            else:
                item = deepcopy(data)
                if stage == "extract":
                    item["preembed_workers"] = config.extraction.workers
                    item["vector_backend"] = "qdrant"
                if stage == "judge" and not item.get("run_id"):
                    raise ConfigError("Judge plans require an existing run_id; select the saved launch manifest.")
                if not item.get("run_id"):
                    item["run_id"] = f"{stage}-{config.label}-{stamp}"
                runs.append(item)
        elif family == "throughput":
            from benchmarks.memory.throughput.config import ThroughputConfig
            ThroughputConfig.from_dict(data, runtime=runtime)
            item = deepcopy(data)
            sweep_name = plan.get("sweep")
            sweep = data.get("sweeps", {}).get(sweep_name, {}) if sweep_name else {}
            if sweep_name and not sweep:
                raise ConfigError(f"No throughput sweep {sweep_name!r} in {path}.")
            if "conditions" in sweep:
                item["conditions"] = sweep["conditions"]
            for source, target in (("kv_store_backend", "store_backend"), ("kv_staging_slots", "staging_slots")):
                if source in sweep:
                    item["inferscale"].setdefault("kv", {})[target] = sweep[source]
            prefix = sweep.get("run_prefix", "throughput")
            if not item.get("run_id"):
                item["run_id"] = f"{prefix}-{stamp}-{runtime.model_label(data['model'])}"
            ThroughputConfig.from_dict(item, runtime=runtime)
            runs.append(item)
        else:
            from benchmarks.rag.config import RagBenchConfig
            if "model" in data and "models" in data:
                raise ConfigError(f"{path}: specify model or models, not both.")
            models = data.get("models", [optional(data, "model", str, "llama", str(path))])
            if not isinstance(models, list) or not models or any(not isinstance(model, str) for model in models):
                raise ConfigError(f"{path}: models must be a non-empty list of model names.")
            if len(models) != len(set(models)):
                raise ConfigError(f"{path}: models must not contain duplicates.")
            sweep_name = plan.get("sweep") if stage == "run" else None
            for model in models:
                model_data = deepcopy(data)
                model_data.pop("model", None)
                model_data["models"] = [model]
                config = RagBenchConfig.from_dict(model_data, runtime=runtime)
                sweeps = data.get("sweeps") or {}
                if sweep_name and sweep_name not in sweeps:
                    raise ConfigError(f"No RAG sweep {sweep_name!r} in {path}.")
                sweep = sweeps[sweep_name] if sweep_name else {}
                if stage == "judge" and not data.get("run_id"):
                    raise ConfigError("Judge plans require an existing run_id; select the saved launch manifest.")
                for top_k in sweep.get("top_k") or [config.top_k]:
                    for backend in sweep.get("answer_backends") or [config.answer_backend]:
                        item = deepcopy(model_data)
                        item["top_k"] = top_k
                        item["answer_backend"] = backend
                        if stage == "run" and sweep_name:
                            item["skip_judge"] = True
                        marker = "kv" if backend == "kv-injection" else "prefix"
                        if item.get("run_id") is None:
                            item["run_id"] = f"{config.dataset_name}-{runtime.model_label(model)}-{marker}-k{top_k}-s{config.context_window}-{stamp}"
                        RagBenchConfig.from_dict(item, runtime=runtime)
                        runs.append(item)
    ids = [run["run_id"] for run in runs]
    if any(not isinstance(item, str) or not item or slug(item) != item.lower() for item in ids):
        raise ConfigError("Run IDs must contain only letters, numbers, dots, dashes and underscores.")
    if len(ids) != len(set(ids)):
        raise ConfigError("Launch plan produces duplicate run IDs; select unique configs or remove fixed run_id values.")
    for run in runs:
        validate_run(run, runtime, family, stage)
    return runs


def materialize_plan(path: Path, runtime: RuntimeConfig, stage: str) -> tuple[dict[str, Any], Path]:
    plan = load_launch_plan(path)
    validate_plan(plan)
    if "manifest" in plan:
        saved_path = expand_path(plan["manifest"], root=runtime.root)
        saved = load_launch_plan(saved_path)
        validate_plan(saved)
        if "run_configs" not in saved or saved["family"] != plan["family"]:
            raise ConfigError("launch.manifest must reference a saved manifest for the same benchmark family.")
        return materialize_plan(saved_path, runtime, stage)
    if "run_configs" in plan:
        configs = plan["run_configs"]
        if not isinstance(configs, list) or not configs or any(not isinstance(item, str) for item in configs):
            raise ConfigError("Saved launch manifest run_configs must be a non-empty list of paths.")
        for item in configs:
            if not Path(item).is_file():
                raise ConfigError(f"Saved run config not found: {item}")
        return plan, path.resolve()
    stamp = plan.get("stamp") or stamp_now()
    if not isinstance(stamp, str) or slug(stamp) != stamp.lower():
        raise ConfigError("launch.stamp must contain only letters, numbers, dots, dashes and underscores.")
    runs = expand_runs(plan, runtime, stage, stamp)
    if stage == "extract":
        for run in runs:
            extraction_command(run, runtime)
    name = plan.get("name") or plan["family"]
    folder = runtime.layout.results_root / "launches" / f"{slug(name)}-{stage}-{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    runtime_path = folder / "runtime.json"
    atomic_write_json(runtime_path, load_json_object(runtime.path), indent=2)
    manifest = {"family": plan["family"], "name": name, "stage": stage,
                "runtime_config": str(runtime_path), "run_configs": []}
    for run in runs:
        # Keep answer models and data/results locations stable during a later
        # judge pass even when it selects a different runtime file.
        if plan["family"] == "memory":
            from benchmarks.memory.config import MemoryRunConfig
            resolved = MemoryRunConfig.from_dict(run, runtime=runtime)
            run["model"] = resolved.model
            run["dataset_path"] = str(resolved.dataset_path)
            run["results_dir"] = str(resolved.results_dir)
        elif plan["family"] == "rag":
            from benchmarks.rag.config import RagBenchConfig
            resolved = RagBenchConfig.from_dict(run, runtime=runtime)
            run.pop("model", None)
            run["models"] = [resolved.model]
            run["data_dir"] = str(resolved.data_dir)
            run["results_dir"] = str(resolved.results_dir)
        else:
            run["model"] = runtime.resolve_model(run["model"])
            run["dataset_path"] = str(expand_path(run.get("dataset_path", "data/locomo10.json"), root=runtime.root))
            run["results_dir"] = str(expand_path(run["results_dir"], root=runtime.root)) if run.get("results_dir") else str(runtime.layout.results_root)
        target = folder / "runs" / f"{run['run_id']}.json"
        atomic_write_json(target, run, indent=2)
        manifest["run_configs"].append(str(target))
    manifest_path = folder / "manifest.json"
    atomic_write_json(manifest_path, manifest, indent=2)
    return manifest, manifest_path


def run_job(job: dict[str, Any], log_file: Path) -> int:
    """Run an isolated benchmark process using a JSON job on standard input."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # A temporary input stream avoids pipe deadlocks while output is streamed.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as source, log_file.open("w", encoding="utf-8") as log:
        json.dump(job, source)
        source.seek(0)
        process = subprocess.Popen(
            [sys.executable, "-m", "benchmarks.common.job"],
            stdin=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def _server_alive(url: str, api_key: str | None = None) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urlopen(Request(url.rstrip("/") + "/models", headers=headers), timeout=3):
            return True
    except (URLError, TimeoutError):
        return False


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def extraction_command(data: dict[str, Any], runtime: RuntimeConfig) -> tuple[list[str], Any]:
    from benchmarks.memory.config import MemoryRunConfig
    from benchmarks.memory.protocol import MEMORY_EXTRACTION_MAX_MODEL_LEN
    config = MemoryRunConfig.from_dict(data, runtime=runtime)
    endpoint = urlsplit(config.memory_llm_base_url)
    if endpoint.hostname not in {"localhost", "127.0.0.1", "::1"} or endpoint.scheme != "http":
        raise ConfigError("Local extraction requires mem0.llm_base_url to be a local HTTP endpoint.")
    if endpoint.port != config.extraction.port or endpoint.path.rstrip("/") != "/v1":
        raise ConfigError("mem0.llm_base_url must match extraction.port and end in /v1.")
    reserved = {"--host", "--port", "--model", "--max-model-len", "--gpu-memory-utilization", "--dtype", "--api-key", "--reasoning-parser"}
    if any(value.split("=", 1)[0] in reserved for value in config.extraction.extra_vllm_args):
        raise ConfigError("extraction.extra_vllm_args cannot override the configured model, endpoint, credentials or server settings.")
    command = ["vllm", "serve", config.model, "--host", endpoint.hostname, "--port", str(config.extraction.port),
               "--trust-remote-code", "--dtype", "auto", "--max-model-len", str(MEMORY_EXTRACTION_MAX_MODEL_LEN),
               "--gpu-memory-utilization", str(config.extraction.gpu_memory_utilization)]
    if config.extraction.structured_outputs:
        command += ["--structured-outputs-config", json.dumps(config.extraction.structured_outputs)]
    parser = runtime.reasoning_parser(data["model"])
    if parser:
        command += ["--reasoning-parser", parser]
    command.extend(config.extraction.extra_vllm_args)
    return command, config


def run_extraction(data: dict[str, Any], runtime: RuntimeConfig, job: dict[str, Any], log_file: Path) -> int:
    server_command, config = extraction_command(data, runtime)
    if _server_alive(config.memory_llm_base_url, config.memory_llm_api_key):
        raise ConfigError(f"An existing server is already using {config.memory_llm_base_url}.")
    server_log = log_file.with_suffix(".server.log")
    server_log.parent.mkdir(parents=True, exist_ok=True)
    if config.memory_llm_api_key:
        server_command += ["--api-key", config.memory_llm_api_key]
    with server_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + config.extraction.health_timeout_s
            while not _server_alive(config.memory_llm_base_url, config.memory_llm_api_key):
                if process.poll() is not None:
                    raise RuntimeError(f"Extraction server exited; see {server_log}.")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"Extraction server health timeout; see {server_log}.")
                time.sleep(2)
            return run_job(job, log_file)
        finally:
            stop_server(process)


def prepare_rag(data: dict[str, Any], runtime: RuntimeConfig) -> None:
    from benchmarks.rag.config import RagBenchConfig
    from benchmarks.rag.datasets import get_dataset
    config = RagBenchConfig.from_dict(data, runtime=runtime)
    spec = get_dataset(config.dataset_name)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in spec.download_urls.items():
        if filename.endswith((".tgz", ".tar.gz")):
            target = config.data_dir / spec.corpus_filename
            if target.exists():
                continue
            with tempfile.TemporaryDirectory(dir=config.data_dir) as temporary:
                archive = Path(temporary) / "dataset.tgz"
                _download(url, archive)
                with tarfile.open(archive) as bundle:
                    member = next((entry for entry in bundle.getmembers() if Path(entry.name).name == spec.corpus_filename and entry.isfile()), None)
                    if member is None:
                        raise ConfigError(f"Dataset archive does not contain {spec.corpus_filename}.")
                    contents = bundle.extractfile(member)
                    assert contents is not None
                    payload = contents.read()
                    atomic_write_json(target, json.loads(payload))
        else:
            target = config.data_dir / filename
            if not target.exists():
                _download(url, target)
    docs, queries = spec.load(config.data_dir)
    print(f"{spec.name}: {len(docs)} documents, {len(queries)} queries")


def _download(url: str, target: Path) -> None:
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    if target.suffix == ".json":
        atomic_write_json(target, json.loads(payload))
    else:
        target.write_bytes(payload)


def execute_manifest(manifest: dict[str, Any], path: Path, runtime: RuntimeConfig, stage: str, *, dry_run: bool) -> int:
    family = manifest["family"]
    validate_stage(family, stage)
    # Validate the complete saved plan before launching any model or server.
    for config_path in manifest["run_configs"]:
        data = load_json_object(config_path)
        validate_run(data, runtime, family, stage)
        if stage == "extract":
            extraction_command(data, runtime)
    failures: list[str] = []
    for index, config_path in enumerate(manifest["run_configs"], 1):
        data = load_json_object(config_path)
        if stage == "judge":
            data["skip_judge"] = False
            judge_path = path.parent / "judge-inputs" / Path(config_path).name
            atomic_write_json(judge_path, data, indent=2)
            config_path = str(judge_path)
        actual_stage = "preembed" if stage == "extract" else stage
        job = {"family": family, "config": config_path, "runtime_config": str(runtime.path), "stage": actual_stage}
        print(f"[{index}/{len(manifest['run_configs'])}] {data['run_id']}", flush=True)
        if stage == "extract":
            server_command, _ = extraction_command(data, runtime)
            print(shlex.join(server_command), flush=True)
        if stage == "prepare":
            if family != "rag":
                raise ConfigError("prepare is supported only for RAG launch plans.")
            from benchmarks.rag.config import RagBenchConfig
            config = RagBenchConfig.from_dict(data, runtime=runtime)
            print(f"Prepare dataset {config.dataset_name} in {config.data_dir}")
        else:
            print(json.dumps(job), flush=True)
        if dry_run:
            continue
        log_file = path.parent / "logs" / f"{data['run_id']}.{stage}.log"
        if stage == "prepare":
            prepare_rag(data, runtime)
            continue
        if stage == "run" and family == "memory":
            guard = {**job, "stage": "check-catalogs"}
            if run_job(guard, log_file.with_suffix(".catalogs.log")):
                failures.append(data["run_id"])
                continue
        code = run_extraction(data, runtime, job, log_file) if stage == "extract" else run_job(job, log_file)
        if code:
            failures.append(data["run_id"])
    if failures:
        print(f"Failed runs: {', '.join(failures)}", file=sys.stderr)
    print(f"Launch manifest: {path}")
    return int(bool(failures))


def launch(config_path: Path) -> int:
    """Execute the stage, runtime and dry-run policy authored in a JSON plan."""
    from benchmarks.common.environment import runtime_path_for_config

    config_path = Path(config_path)
    source = load_launch_plan(config_path)
    validate_plan(source)
    stage = source.get("stage") or "run"
    validate_stage(source["family"], stage)
    dry_run = optional(source, "dry_run", bool, False, "launch")
    runtime = load_runtime_config(runtime_path_for_config(config_path))
    runtime.apply_environment()
    if not dry_run:
        runtime.layout.create_directories()
    manifest, path = materialize_plan(config_path, runtime, stage)
    if "run_configs" not in source and "manifest" not in source:
        runtime = load_runtime_config(manifest["runtime_config"])
    return execute_manifest(manifest, path, runtime, stage, dry_run=dry_run)
