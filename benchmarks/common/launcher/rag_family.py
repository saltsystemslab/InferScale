"""RAG benchmark runs: one run per model, retrieval depth, and answer backend."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError, optional
from benchmarks.rag.config import LMCACHE_BACKEND, RagBenchConfig, validate_rag_stage

RUN_ID_MARKERS = {
    "kv-injection": "kv",
    "prompt-injection": "prompt-injection",
    LMCACHE_BACKEND: "lmcache-blend",
}


def expand(
    data: dict[str, Any],
    plan: dict[str, Any],
    runtime: RuntimeConfig,
    stage: str,
    stamp: str,
    where: str,
) -> list[dict[str, Any]]:
    models = _models(data, where)
    sweep_name = plan.get("sweep") if stage == "run" else None
    sweep = _sweep(data, sweep_name, where)
    if stage == "judge" and not data.get("run_id"):
        raise ConfigError(
            "Judge plans require an existing run_id; select the saved launch manifest."
        )
    runs = []
    for model in models:
        model_data = deepcopy(data)
        model_data.pop("model", None)
        model_data["models"] = [model]
        config = RagBenchConfig.from_dict(model_data, runtime=runtime)
        for top_k in sweep.get("top_k") or [config.top_k]:
            for backend in sweep.get("answer_backends") or [config.answer_backend]:
                run = deepcopy(model_data)
                run["top_k"] = top_k
                run["answer_backend"] = backend
                if stage == "run" and sweep_name:
                    run["skip_judge"] = True
                if run.get("run_id") is None:
                    run["run_id"] = _run_id(config, runtime, model, top_k, backend, stamp)
                RagBenchConfig.from_dict(run, runtime=runtime)
                runs.append(run)
    return runs


def _models(data: dict[str, Any], where: str) -> list[str]:
    if "model" in data and "models" in data:
        raise ConfigError(f"{where}: specify model or models, not both.")
    models = data.get("models", [optional(data, "model", str, "llama", where)])
    if not isinstance(models, list) or not models or any(not isinstance(m, str) for m in models):
        raise ConfigError(f"{where}: models must be a non-empty list of model names.")
    if len(models) != len(set(models)):
        raise ConfigError(f"{where}: models must not contain duplicates.")
    return models


def _sweep(data: dict[str, Any], sweep_name: str | None, where: str) -> dict[str, Any]:
    if not sweep_name:
        return {}
    sweeps = data.get("sweeps") or {}
    if sweep_name not in sweeps:
        raise ConfigError(f"No RAG sweep {sweep_name!r} in {where}.")
    return sweeps[sweep_name]


def _run_id(
    config: RagBenchConfig, runtime: RuntimeConfig, model: str, top_k: int, backend: str, stamp: str
) -> str:
    marker = RUN_ID_MARKERS.get(backend, backend)
    return (
        f"{config.dataset_name}-{runtime.model_label(model)}-{marker}"
        f"-k{top_k}-s{config.context_window}-{stamp}"
    )


def validate_run(data: dict[str, Any], runtime: RuntimeConfig, stage: str) -> None:
    config = RagBenchConfig.from_dict(data, runtime=runtime)
    validate_rag_stage(config, stage, has_run_id=True)


def pin_resolved_paths(run: dict[str, Any], runtime: RuntimeConfig) -> None:
    resolved = RagBenchConfig.from_dict(run, runtime=runtime)
    run.pop("model", None)
    run["models"] = [resolved.model]
    run["data_dir"] = str(resolved.data_dir)
    run["results_dir"] = str(resolved.results_dir)
