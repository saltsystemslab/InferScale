from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from loguru import logger

JOB_COMMAND = [sys.executable, "-m", "benchmarks.common.job"]
JOB_STOP_GRACE_PERIOD_S = 30
SERVER_STOP_GRACE_PERIOD_S = 60
HEALTH_CHECK_TIMEOUT_S = 3


def run_job(job: dict[str, Any], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with _job_as_input_file(job) as job_input, log_file.open("w", encoding="utf-8") as job_log:
        process = subprocess.Popen(
            JOB_COMMAND,
            stdin=job_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _relay_output(process, job_log)
            return process.wait()
        finally:
            _stop_job(process)


@contextlib.contextmanager
def _job_as_input_file(job: dict[str, Any]) -> Iterator[IO[str]]:
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as job_file:
        json.dump(job, job_file)
        job_file.seek(0)
        yield job_file


def _relay_output(process: subprocess.Popen[str], job_log: IO[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        logger.opt(raw=True).info(line)
        job_log.write(line)


def _stop_job(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=JOB_STOP_GRACE_PERIOD_S)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def start_server(command: list[str], output: IO[str]) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
    )


def stop_server(server: subprocess.Popen[Any]) -> None:
    if server.poll() is not None:
        return
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        server.wait()
        return
    try:
        server.wait(timeout=SERVER_STOP_GRACE_PERIOD_S)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def server_alive(url: str, api_key: str | None = None) -> bool:
    models_url = url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with urlopen(Request(models_url, headers=headers), timeout=HEALTH_CHECK_TIMEOUT_S):
            return True
    except (URLError, TimeoutError):
        return False
