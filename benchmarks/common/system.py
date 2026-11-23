from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.common.paths import project_root

PACKAGES = ("torch", "vllm", "lmcache", "mem0ai", "openai", "jasper", "qdrant-client")
NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,driver_version,memory.total",
    "--format=csv,noheader",
]
NVIDIA_SMI_TIMEOUT_S = 5
GPU_FIELDS = ("name", "driver_version", "memory_total")
MEMINFO_PATH = Path("/proc/meminfo")
BYTES_PER_MEMINFO_KB = 1024
ASSUMED_AVAILABLE_FRACTION = 0.8


def collect_system_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in PACKAGES},
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
        nvidia_smi = subprocess.run(
            NVIDIA_SMI_QUERY,
            check=False,
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "gpus": []}
    if nvidia_smi.returncode != 0:
        return {"available": False, "gpus": [], "error": nvidia_smi.stderr.strip()}
    gpus = []
    for line in nvidia_smi.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= len(GPU_FIELDS):
            gpus.append(dict(zip(GPU_FIELDS, fields)))
    return {"available": bool(gpus), "count": len(gpus), "gpus": gpus}


def _jasper_commit() -> str | None:
    jasper_dir = project_root() / "jasperpy"
    if not jasper_dir.exists():
        return None
    try:
        rev_parse = subprocess.run(
            ["git", "-C", str(jasper_dir), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    if rev_parse.returncode != 0:
        return None
    return rev_parse.stdout.strip()


def available_host_memory_bytes() -> int | None:
    reported_by_kernel = _meminfo_available_bytes()
    if reported_by_kernel is not None:
        return reported_by_kernel
    physical_bytes = total_host_memory_bytes()
    if physical_bytes is None:
        return None
    return int(physical_bytes * ASSUMED_AVAILABLE_FRACTION)


def _meminfo_available_bytes() -> int | None:
    try:
        return parse_mem_available_bytes(MEMINFO_PATH.read_text(encoding="ascii"))
    except OSError:
        return None


def total_host_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return int(page_size) * int(page_count)


def parse_mem_available_bytes(meminfo_text: str) -> int | None:
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                return int(fields[1]) * BYTES_PER_MEMINFO_KB
    return None
