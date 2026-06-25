from __future__ import annotations

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
    default_user_id: str = "default",
) -> dict[str, Any]:
    return {
        "kv_connector": "MemoryKVConnector",
        "kv_role": "kv_both",
        "kv_connector_module_path": connector_module,
        "kv_connector_extra_config": {
            "memory_namespace": namespace,
            "default_user_id": default_user_id,
        },
    }


def force_vllm_inprocess_mode() -> None:
    """Force vLLM V1 offline LLM execution into this process."""
    sanitize_repo_vllm_env_for_import()
    current = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if _env_truthy(current):
        logger.warning(
            "Overriding VLLM_ENABLE_V1_MULTIPROCESSING=%s because strict GPU KV mode requires "
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


def synchronize_cuda() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}
