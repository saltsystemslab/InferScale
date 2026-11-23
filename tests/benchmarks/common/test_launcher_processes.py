from __future__ import annotations

import itertools
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest
from loguru import logger

from benchmarks.common.config import ConfigError, load_json_object
from benchmarks.common.launcher import extraction, processes

ROOT = Path(__file__).resolve().parents[3]
MEMORY_CONFIG = ROOT / "configs/memory/accuracy-latency/llama.json"
PREEMBED_JOB = {"family": "memory", "stage": "preembed"}


class OpenedUrl:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_server_alive_probes_the_models_endpoint_with_the_api_key(monkeypatch):
    probes = []

    def urlopen(request, timeout):
        probes.append((request.full_url, request.get_header("Authorization"), timeout))
        return OpenedUrl()

    monkeypatch.setattr(processes, "urlopen", urlopen)

    assert processes.server_alive("http://localhost:8000/v1/", "secret-key")
    assert processes.server_alive("http://localhost:8000/v1")

    assert probes == [
        ("http://localhost:8000/v1/models", "Bearer secret-key", 3),
        ("http://localhost:8000/v1/models", None, 3),
    ]


@pytest.mark.parametrize("failure", [URLError("connection refused"), TimeoutError()])
def test_server_alive_is_false_while_the_endpoint_does_not_answer(monkeypatch, failure):
    def urlopen(request, timeout):
        raise failure

    monkeypatch.setattr(processes, "urlopen", urlopen)

    assert not processes.server_alive("http://localhost:8000/v1")


class ServerProcess:
    pid = 99

    def __init__(self, *, exit_code=None, stops_within_grace_period=True):
        self.exit_code = exit_code
        self.stops_within_grace_period = stops_within_grace_period
        self.waits = []

    def poll(self):
        return self.exit_code

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if timeout is not None and not self.stops_within_grace_period:
            raise subprocess.TimeoutExpired("vllm", timeout)


def record_group_signals(monkeypatch, *, group_exists=True):
    signals = []

    def killpg(pid, sent):
        signals.append((pid, sent))
        if not group_exists:
            raise ProcessLookupError

    monkeypatch.setattr(processes.os, "killpg", killpg)
    return signals


def test_stop_server_leaves_an_exited_server_alone(monkeypatch):
    signals = record_group_signals(monkeypatch)
    server = ServerProcess(exit_code=0)

    processes.stop_server(server)

    assert signals == []
    assert server.waits == []


def test_stop_server_only_asks_a_cooperative_server_to_terminate(monkeypatch):
    signals = record_group_signals(monkeypatch)
    server = ServerProcess()

    processes.stop_server(server)

    assert signals == [(99, signal.SIGTERM)]
    assert server.waits == [60]


def test_stop_server_kills_a_server_that_outlives_the_grace_period(monkeypatch):
    signals = record_group_signals(monkeypatch)
    server = ServerProcess(stops_within_grace_period=False)

    processes.stop_server(server)

    assert signals == [(99, signal.SIGTERM), (99, signal.SIGKILL)]
    assert server.waits == [60, None]


def test_stop_server_reaps_a_server_whose_process_group_is_already_gone(monkeypatch):
    signals = record_group_signals(monkeypatch, group_exists=False)
    server = ServerProcess()

    processes.stop_server(server)

    assert signals == [(99, signal.SIGTERM)]
    assert server.waits == [None]


class FinishedJob:
    def __init__(self, output, exit_code):
        self.stdout = iter(output)
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code

    def wait(self, timeout=None):
        return self.exit_code


def test_job_output_reaches_the_launcher_log_without_a_second_log_prefix(tmp_path, monkeypatch):
    output = ["2026-11-23 12:00:00 | INFO  | loaded {model}\n", "plain line\n"]
    started = []

    def popen(command, **options):
        started.append((command, json.load(options["stdin"]), options["stderr"]))
        return FinishedJob(output, exit_code=3)

    monkeypatch.setattr(processes.subprocess, "Popen", popen)
    relayed = []
    sink = logger.add(relayed.append, format="LAUNCHER | {message}", level="INFO")
    try:
        exit_code = processes.run_job(PREEMBED_JOB, tmp_path / "job.log")
    finally:
        logger.remove(sink)

    assert exit_code == 3
    assert relayed == output
    assert (tmp_path / "job.log").read_text() == "".join(output)
    assert started == [
        ([sys.executable, "-m", "benchmarks.common.job"], PREEMBED_JOB, subprocess.STDOUT)
    ]


class InterruptedJob:
    def __init__(self, *, stops_within_grace_period):
        self.stops_within_grace_period = stops_within_grace_period
        self.stdout = self.output()
        self.events = []

    def output(self):
        yield "first line\n"
        raise KeyboardInterrupt

    def poll(self):
        return None

    def terminate(self):
        self.events.append("terminate")

    def kill(self):
        self.events.append("kill")

    def wait(self, timeout=None):
        if timeout is not None and not self.stops_within_grace_period:
            raise subprocess.TimeoutExpired("job", timeout)
        self.events.append(("wait", timeout))


@pytest.mark.parametrize(
    "stops_within_grace_period,expected_events",
    [
        (True, ["terminate", ("wait", 30)]),
        (False, ["terminate", "kill", ("wait", None)]),
    ],
)
def test_interrupted_job_is_terminated_and_killed_only_if_it_lingers(
    tmp_path, monkeypatch, launcher_log, stops_within_grace_period, expected_events
):
    job = InterruptedJob(stops_within_grace_period=stops_within_grace_period)
    monkeypatch.setattr(processes.subprocess, "Popen", lambda *args, **kwargs: job)
    log_file = tmp_path / "logs" / "job.log"

    with pytest.raises(KeyboardInterrupt):
        processes.run_job(PREEMBED_JOB, log_file)

    assert job.events == expected_events
    assert log_file.read_text() == "first line\n"
    assert launcher_log == ["first line\n"]


class ExtractionServer:
    def __init__(self, monkeypatch, *, answers_health_checks, exit_code=None):
        self.started = []
        self.stopped = []
        self.health_checks = []
        self.process = SimpleNamespace(pid=123, poll=lambda: exit_code)
        monkeypatch.setattr(extraction.subprocess, "Popen", self.start)
        monkeypatch.setattr(extraction, "stop_server", self.stopped.append)
        monkeypatch.setattr(extraction, "server_alive", self.alive)
        self.answers = iter(answers_health_checks)

    def start(self, command, **options):
        self.started.append((list(command), options))
        return self.process

    def alive(self, url, api_key):
        self.health_checks.append((url, api_key))
        return next(self.answers)


def test_extraction_refuses_to_start_beside_a_running_server(runtime, tmp_path, monkeypatch):
    server = ExtractionServer(monkeypatch, answers_health_checks=[True])
    monkeypatch.setattr(extraction, "run_job", lambda *_: pytest.fail("Job reused a server"))

    with pytest.raises(ConfigError, match="already using http://localhost:8000/v1"):
        extraction.run_extraction(
            load_json_object(MEMORY_CONFIG), runtime, PREEMBED_JOB, tmp_path / "extract.log"
        )

    assert server.started == []
    assert server.stopped == []


def test_extraction_server_gets_the_api_key_that_its_announced_command_omits(
    runtime, tmp_path, monkeypatch
):
    monkeypatch.setenv("EXTRACTION_LLM_API_KEY", "secret-key")
    data = load_json_object(MEMORY_CONFIG)
    announced_command, _ = extraction.extraction_command(data, runtime)
    server = ExtractionServer(monkeypatch, answers_health_checks=[False, False, True])
    monkeypatch.setattr(extraction.time, "sleep", lambda seconds: None)
    jobs = []

    def run_job(job, log_file):
        jobs.append((job, log_file))
        return 7

    monkeypatch.setattr(extraction, "run_job", run_job)
    log_file = tmp_path / "logs" / "extract.log"

    assert extraction.run_extraction(data, runtime, PREEMBED_JOB, log_file) == 7

    assert "secret-key" not in announced_command
    [(command, options)] = server.started
    assert command == [*announced_command, "--api-key", "secret-key"]
    assert options["start_new_session"] is True
    assert options["stderr"] == subprocess.STDOUT
    assert options["stdout"].name == str(tmp_path / "logs" / "extract.server.log")
    assert server.health_checks == [("http://localhost:8000/v1", "secret-key")] * 3
    assert jobs == [(PREEMBED_JOB, log_file)]
    assert server.stopped == [server.process]


def test_extraction_fails_when_its_server_exits_before_serving(runtime, tmp_path, monkeypatch):
    server = ExtractionServer(monkeypatch, answers_health_checks=[False, False], exit_code=1)
    monkeypatch.setattr(extraction, "run_job", lambda *_: pytest.fail("Job ran without a server"))
    log_file = tmp_path / "extract.log"

    with pytest.raises(RuntimeError, match="Extraction server exited; see .*extract.server.log"):
        extraction.run_extraction(load_json_object(MEMORY_CONFIG), runtime, PREEMBED_JOB, log_file)

    assert server.stopped == [server.process]


def test_extraction_fails_when_its_server_is_not_healthy_in_time(runtime, tmp_path, monkeypatch):
    data = load_json_object(MEMORY_CONFIG)
    data["extraction"] = {**data.get("extraction", {}), "health_timeout_s": 5}
    server = ExtractionServer(monkeypatch, answers_health_checks=itertools.repeat(False))
    clock = itertools.count(start=0, step=2)
    naps = []
    monkeypatch.setattr(extraction.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(extraction.time, "sleep", naps.append)
    monkeypatch.setattr(extraction, "run_job", lambda *_: pytest.fail("Job ran without a server"))

    with pytest.raises(RuntimeError, match="Extraction server health timeout"):
        extraction.run_extraction(data, runtime, PREEMBED_JOB, tmp_path / "extract.log")

    assert naps == [2, 2]
    assert server.stopped == [server.process]
