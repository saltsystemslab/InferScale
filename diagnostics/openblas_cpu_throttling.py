# Tests whether OpenBLAS worker threads drain the container's CPU quota after a local Qdrant
# search, and whether that is what inflates engine TTFT in the Qdrant arms.
#
# The hypothesis:
# 1. Where OpenBLAS comes from: qdrant-client in local mode scores every stored vector with
#    numpy.dot, and numpy's wheel bundles OpenBLAS, which runs that product on one worker
#    thread per visible CPU (up to 64).
# 2. After the product the idle workers do not sleep.
#    They spin for about 100 ms waiting for more work, each one using a full core.
# 3. The container has a CPU quota: a budget of CPU time per 100 ms period, shared by all of
#    its threads.
#    A quota of 8 CPUs is 800 ms of CPU time per period, which 64 spinning workers use up
#    (drain) in about 12 ms.
# 4. Once the budget is drained, the kernel throttles the container: it stops every thread in
#    it until the next period starts, which can be most of the 100 ms away.
# 5. The stopped threads include the one running vLLM, so a generate() call made right after
#    the search is frozen with everything else.
#    vLLM then reports a time to first token that is tens of milliseconds longer, although the
#    engine did the same work.
#
# Each trial runs a workload, then watches the main thread for pauses.
# The script reruns itself once per BLAS setting, because OpenBLAS fixes its thread count when
# numpy loads.
# If the hypothesis holds, only "OpenBLAS default" after "Qdrant search" shows frozen trials,
# busy cores above 1 (idle workers spinning) and growing throttled periods (the kernel's
# nr_throttled counter).
#
# Run it while nothing else is using the machine:
#   python diagnostics/openblas_cpu_throttling.py

from __future__ import annotations

import os
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient, models

PARAMETERS = {
    "blas_settings": {
        "OpenBLAS default": {},
        "OPENBLAS_NUM_THREADS=1": {"OPENBLAS_NUM_THREADS": "1"},
    },
    "trials_per_row": 20,
    "stored_vectors": 2000,
    "embedding_dimensions": 1536,
    "top_k": 5,
    "idle_seconds_before_trial": (0.3, 0.4),
    "watch_seconds": 0.3,
    "frozen_pause_ms": 20,
}
OPENBLAS_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS", "OMP_NUM_THREADS")
COLUMN_WIDTHS = {
    "BLAS setting": 24,
    "workload": 15,
    "threads": 9,
    "busy cores": 12,
    "frozen trials": 15,
    "median pause": 14,
    "longest pause": 15,
    "throttled periods": 17,
}


@dataclass(frozen=True)
class Watch:
    longest_pause_ms: float
    busy_cores: float


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, format="{message}")
    if len(sys.argv) > 1:
        measure(blas_setting=sys.argv[1])
    else:
        compare_blas_settings()


def compare_blas_settings() -> None:
    log_machine()
    logger.info("".join(column.ljust(width) for column, width in COLUMN_WIDTHS.items()))
    for blas_setting, thread_variables in PARAMETERS["blas_settings"].items():
        measure_in_fresh_process(blas_setting, thread_variables)


def log_machine() -> None:
    quota = cpu_quota()
    thread_limits = [
        f"{name}={os.environ[name]}" for name in OPENBLAS_THREAD_VARIABLES if name in os.environ
    ]
    logger.info(
        "CPUs available: {} | CPU quota: {} | thread limits in this shell: {}",
        available_cpu_count(),
        "none visible" if quota is None else f"{quota:g} CPUs",
        ", ".join(thread_limits) or "none",
    )
    logger.info(
        "Qdrant store: {} x {} | trials per row: {} | watch: {:.0f} ms",
        PARAMETERS["stored_vectors"],
        PARAMETERS["embedding_dimensions"],
        PARAMETERS["trials_per_row"],
        PARAMETERS["watch_seconds"] * 1000,
    )


def measure_in_fresh_process(blas_setting: str, thread_variables: dict[str, str]) -> None:
    without_thread_limits = {
        name: value for name, value in os.environ.items() if name not in OPENBLAS_THREAD_VARIABLES
    }
    subprocess.run(
        [sys.executable, __file__, blas_setting],
        env={**without_thread_limits, **thread_variables},
        check=True,
    )


def measure(blas_setting: str) -> None:
    workloads = {"none": do_nothing, "Qdrant search": qdrant_search()}
    threads = thread_count()
    for workload_name, workload in workloads.items():
        throttled_before = throttled_periods()
        watches = [watch_after(workload) for _ in range(PARAMETERS["trials_per_row"])]
        throttled = None if throttled_before is None else throttled_periods() - throttled_before
        pauses = [watch.longest_pause_ms for watch in watches]
        frozen = sum(pause >= PARAMETERS["frozen_pause_ms"] for pause in pauses)
        row = {
            "BLAS setting": blas_setting,
            "workload": workload_name,
            "threads": shown(threads),
            "busy cores": f"{statistics.median(watch.busy_cores for watch in watches):.1f}",
            "frozen trials": f"{frozen}/{len(watches)}",
            "median pause": f"{statistics.median(pauses):.1f} ms",
            "longest pause": f"{max(pauses):.1f} ms",
            "throttled periods": shown(throttled),
        }
        cells = [row[column].ljust(width) for column, width in COLUMN_WIDTHS.items()]
        logger.info("".join(cells).rstrip())


def do_nothing() -> None:
    pass


def qdrant_search() -> Callable[[], object]:
    collection = "memories"
    dimensions = PARAMETERS["embedding_dimensions"]
    embeddings = np.random.default_rng(0).standard_normal(
        (PARAMETERS["stored_vectors"] + 1, dimensions), dtype=np.float32
    )
    client = QdrantClient(":memory:")
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.DOT),
    )
    client.upload_collection(collection, vectors=embeddings[1:])
    query = embeddings[0].tolist()
    return lambda: client.query_points(
        collection, query=query, limit=PARAMETERS["top_k"], with_payload=True
    )


def watch_after(workload: Callable[[], object]) -> Watch:
    time.sleep(random.uniform(*PARAMETERS["idle_seconds_before_trial"]))
    workload()
    return watch_main_thread()


def watch_main_thread() -> Watch:
    watch_seconds = PARAMETERS["watch_seconds"]
    cpu_started = time.process_time()
    started = previous = time.perf_counter()
    longest_pause = 0.0
    while previous - started < watch_seconds:
        now = time.perf_counter()
        longest_pause = max(longest_pause, now - previous)
        previous = now
    return Watch(
        longest_pause_ms=longest_pause * 1000,
        busy_cores=(time.process_time() - cpu_started) / (previous - started),
    )


def shown(count: int | None) -> str:
    return "n/a" if count is None else str(count)


def available_cpu_count() -> int | None:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def thread_count() -> int | None:
    threads = Path("/proc/self/task")
    return len(list(threads.iterdir())) if threads.is_dir() else None


def cpu_quota() -> float | None:
    limits = [line.split() for line in cgroup_lines("cpu.max")]
    return min(
        (int(quota) / int(period) for quota, period in limits if quota != "max"), default=None
    )


def throttled_periods() -> int | None:
    counts = [
        int(line.split()[1])
        for line in cgroup_lines("cpu.stat")
        if line.startswith("nr_throttled ")
    ]
    return sum(counts) if counts else None


def cgroup_lines(file_name: str) -> list[str]:
    return [
        line
        for directory in cgroup_directories()
        if (directory / file_name).is_file()
        for line in (directory / file_name).read_text().splitlines()
    ]


def cgroup_directories() -> list[Path]:
    root = Path("/sys/fs/cgroup")
    membership = Path("/proc/self/cgroup")
    if not membership.is_file():
        return []
    unified_hierarchy_paths = [
        line.removeprefix("0::")
        for line in membership.read_text().splitlines()
        if line.startswith("0::")
    ]
    if not unified_hierarchy_paths:
        return []
    own = root / unified_hierarchy_paths[0].lstrip("/")
    return [directory for directory in (own, *own.parents) if directory.is_relative_to(root)]


if __name__ == "__main__":
    main()
