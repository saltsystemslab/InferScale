"""Benchmark jobs as child processes, and the servers some stages start for them."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def run_job(job: dict[str, Any], log_file: Path) -> int:
    """Run one benchmark job in a fresh process, streaming its output to stdout and a log."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # The job is handed over on a temporary file rather than a pipe so streaming
    # the child's output cannot deadlock against an unread stdin.
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as source,
        log_file.open("w", encoding="utf-8") as log,
    ):
        json.dump(job, source)
        source.seek(0)
        process = subprocess.Popen(
            [sys.executable, "-m", "benchmarks.common.job"],
            stdin=source,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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


def server_alive(url: str, api_key: str | None = None) -> bool:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urlopen(Request(url.rstrip("/") + "/models", headers=headers), timeout=3):
            return True
    except (URLError, TimeoutError):
        return False


def stop_server(process: subprocess.Popen[Any]) -> None:
    """Terminate the server's whole process group, escalating to SIGKILL after a minute."""
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
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
