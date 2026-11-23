from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.common import system

MEMINFO = Path("/proc/meminfo")


class Host:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.monkeypatch = monkeypatch

    def memory(self, *, meminfo: Any, page_size: Any = 4096, page_count: Any = 1000) -> None:
        read_any_file = Path.read_text

        def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
            if path != MEMINFO:
                return read_any_file(path, *args, **kwargs)
            if isinstance(meminfo, Exception):
                raise meminfo
            return meminfo

        def sysconf(name: str) -> int:
            return {"SC_PAGE_SIZE": page_size, "SC_PHYS_PAGES": page_count}[name]

        self.monkeypatch.setattr(system.Path, "read_text", read_text)
        self.monkeypatch.setattr(system.os, "sysconf", sysconf)


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> Host:
    return Host(monkeypatch)


def test_system_metadata_has_no_jasper_commit_when_git_is_not_installed(monkeypatch, tmp_path):
    (tmp_path / "jasperpy").mkdir()
    monkeypatch.setattr(system, "project_root", lambda: tmp_path)

    def no_such_program(command, **options):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(system.subprocess, "run", no_such_program)

    assert system.collect_system_metadata()["jasper_commit"] is None


@pytest.mark.parametrize(
    "meminfo",
    [FileNotFoundError("no /proc/meminfo"), "MemTotal: 9000 kB\n"],
)
def test_without_a_kernel_figure_a_fifth_of_physical_ram_is_left_to_kernel_cache_and_processes(
    host, meminfo
):
    host.memory(meminfo=meminfo, page_size=4096, page_count=1001)

    assert system.available_host_memory_bytes() == 3280076
    assert system.total_host_memory_bytes() == 4100096
