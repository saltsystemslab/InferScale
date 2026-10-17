"""Host metadata recorded with every result: platform, package versions, GPUs."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from benchmarks.common.paths import project_root

PACKAGES = ("torch", "vllm", "mem0ai", "openai", "jasper", "qdrant-client")
# Without a MemAvailable figure, this share of the physical RAM is treated as
# available; the kernel, page cache, and resident processes own an unknown share.
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
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
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
    jasper_dir = project_root() / "jasperpy"
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


def available_host_memory_bytes() -> int | None:
    """Host RAM the kernel reports as available, or an estimate from the physical total."""
    try:
        available = parse_mem_available_bytes(Path("/proc/meminfo").read_text(encoding="ascii"))
    except OSError:
        available = None
    if available is not None:
        return available
    total = total_host_memory_bytes()
    return None if total is None else int(total * ASSUMED_AVAILABLE_FRACTION)


def total_host_memory_bytes() -> int | None:
    """Physical host RAM, or None where the platform does not report it."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return int(page_size) * int(page_count)


def parse_mem_available_bytes(meminfo_text: str) -> int | None:
    """The MemAvailable line of /proc/meminfo in bytes, or None when absent or malformed."""
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None
