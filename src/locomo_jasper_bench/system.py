from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def collect_system_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            "torch": _package_version("torch"),
            "vllm": _package_version("vllm"),
            "mem0ai": _package_version("mem0ai"),
            "openai": _package_version("openai"),
            "jasper": _package_version("jasper"),
            "qdrant-client": _package_version("qdrant-client"),
        },
        "gpu": _gpu_metadata(),
        "jasper_commit": _jasper_commit(),
        "vllm_engine": _vllm_engine_metadata(),
    }


def _vllm_engine_metadata() -> dict[str, Any]:
    multiprocessing_env = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    return {
        "vllm_enable_v1_multiprocessing": multiprocessing_env,
        "inprocess": multiprocessing_env == "0",
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_metadata() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "gpus": []}
    if result.returncode != 0:
        return {"available": False, "gpus": [], "error": result.stderr.strip()}
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "driver_version": parts[1], "memory_total": parts[2]})
    return {"available": bool(gpus), "count": len(gpus), "gpus": gpus}


def _jasper_commit() -> str | None:
    jasper_dir = Path("jasperpy")
    if not jasper_dir.exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(jasper_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()
