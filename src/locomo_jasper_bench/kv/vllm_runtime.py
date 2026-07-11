from __future__ import annotations

import gc
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)

_REPO_OWNED_VLLM_ENV_TO_CLEAR = (
    "VLLM_MODEL",
    "VLLM_API_KEY",
    "VLLM_TP",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_DTYPE",
    "VLLM_QUANTIZATION",
    "VLLM_BASE_URL",
)


def build_strict_gpu_kv_transfer_config(
    *,
    connector_module: str,
    namespace: str,
    default_user_id: str | None = "default",
    allow_prefix_scan: bool = False,
    log_memory_hits: bool = True,
) -> dict[str, Any]:
    extra_config: dict[str, Any] = {
        "memory_namespace": namespace,
        "allow_prefix_scan": allow_prefix_scan,
        "log_memory_hits": log_memory_hits,
    }
    if default_user_id is not None:
        extra_config["default_user_id"] = default_user_id
    return {
        "kv_connector": "MemoryKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": connector_module,
        "kv_connector_extra_config": extra_config,
    }


def force_vllm_inprocess_mode() -> None:
    """Force vLLM V1 offline LLM execution into this process.

    Strict GPU KV mode requires the engine in-process, and prefix mode must match
    so both answer backends measure the same engine execution mode.
    """
    sanitize_repo_vllm_env_for_import()
    current = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if _env_truthy(current):
        logger.warning(
            "Overriding VLLM_ENABLE_V1_MULTIPROCESSING=%s because both answer backends require "
            "vLLM's offline engine to share this process.",
            current,
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    loaded_envs = sys.modules.get("vllm.envs")
    if loaded_envs is not None and hasattr(loaded_envs, "VLLM_ENABLE_V1_MULTIPROCESSING"):
        setattr(loaded_envs, "VLLM_ENABLE_V1_MULTIPROCESSING", False)


def sanitize_repo_vllm_env_for_import() -> None:
    """Clear repo convenience env vars that newer vLLM treats as unknown."""
    for name in _REPO_OWNED_VLLM_ENV_TO_CLEAR:
        os.environ.pop(name, None)


def common_vllm_kwargs(config: Any) -> dict[str, Any]:
    return {
        "model": config.model,
        "dtype": config.kv_dtype,
        "trust_remote_code": True,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "swap_space": 0,
        "cpu_offload_gb": 0,
        "gpu_memory_utilization": config.kv_gpu_memory_utilization,
        "block_size": config.kv_block_size,
        "max_model_len": config.kv_max_model_len,
    }


def empty_cuda_cache(*, collect_ipc: bool = False) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        if collect_ipc:
            torch.cuda.ipc_collect()
    except Exception:
        return


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}
