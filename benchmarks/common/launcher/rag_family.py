from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError, optional
from benchmarks.rag.config import LMCACHE_BACKEND, RagBenchConfig, validate_rag_stage

DEFAULT_MODEL = "llama"
RUN_ID_MARKERS = {
    "kv-injection": "kv",
    "prompt-injection": "prompt-injection",
    LMCACHE_BACKEND: "lmcache-blend",
}


def expand(
    authored: dict[str, Any],
    *,
    config_path: str,
    runtime: RuntimeConfig,
    stage: str,
    stamp: str,
    sweep_name: str | None,
) -> list[dict[str, Any]]:
    models = _models(authored, config_path)
    sweep = _sweep(authored, sweep_name, config_path)
    runs = []
    for model in models:
        authored_for_model = _with_only_model(authored, model)
        config = RagBenchConfig.from_dict(authored_for_model, runtime=runtime)
        top_ks = sweep.get("top_k") or [config.top_k]
        answer_backends = sweep.get("answer_backends") or [config.answer_backend]
        for top_k, answer_backend in product(top_ks, answer_backends):
            run = deepcopy(authored_for_model)
            run["top_k"] = top_k
            run["answer_backend"] = answer_backend
            if sweep_name:
                run["skip_judge"] = True
            if run.get("run_id") is None:
                run["run_id"] = _run_id(config, runtime, model, top_k, answer_backend, stamp)
            runs.append(run)
    return runs


def _models(authored: dict[str, Any], config_path: str) -> list[str]:
    if "model" in authored and "models" in authored:
        raise ConfigError(f"{config_path}: specify model or models, not both.")
    if "models" in authored:
        models = authored["models"]
    else:
        models = [optional(authored, "model", str, DEFAULT_MODEL, config_path)]
    if not isinstance(models, list) or not models or any(not isinstance(m, str) for m in models):
        raise ConfigError(f"{config_path}: models must be a non-empty list of model names.")
    if len(models) != len(set(models)):
        raise ConfigError(f"{config_path}: models must not contain duplicates.")
    return models


def _sweep(authored: dict[str, Any], sweep_name: str | None, config_path: str) -> dict[str, Any]:
    if not sweep_name:
        return {}
    sweeps = authored.get("sweeps") or {}
    if sweep_name not in sweeps:
        raise ConfigError(f"No RAG sweep {sweep_name!r} in {config_path}.")
    return sweeps[sweep_name]


def _with_only_model(authored: dict[str, Any], model: str) -> dict[str, Any]:
    authored_for_model = deepcopy(authored)
    authored_for_model.pop("model", None)
    authored_for_model["models"] = [model]
    return authored_for_model


def _run_id(
    config: RagBenchConfig,
    runtime: RuntimeConfig,
    model: str,
    top_k: int,
    answer_backend: str,
    stamp: str,
) -> str:
    marker = RUN_ID_MARKERS.get(answer_backend, answer_backend)
    return (
        f"{config.dataset_name}-{runtime.model_label(model)}-{marker}"
        f"-k{top_k}-s{config.context_window}-{stamp}"
    )


def validate_run(run: dict[str, Any], runtime: RuntimeConfig, stage: str) -> None:
    config = RagBenchConfig.from_dict(run, runtime=runtime)
    validate_rag_stage(config, stage, has_run_id=True)


def pin_to_runtime(run: dict[str, Any], runtime: RuntimeConfig) -> None:
    config = RagBenchConfig.from_dict(run, runtime=runtime)
    run.pop("model", None)
    run["models"] = [config.model]
    run["data_dir"] = str(config.data_dir)
    run["results_dir"] = str(config.results_dir)
