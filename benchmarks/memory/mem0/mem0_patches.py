"""The one Mem0 internal the extraction protocol patches: the prompt's observation date.

Mem0's ``add`` builds the extraction prompt without a timestamp, so the
prompt would carry today's date for every turn of a conversation recorded
years ago. The benchmark wraps the prompt builder once per process and reads
the source turn's date from a context variable, so concurrent extraction
threads each see their own turn's date.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


class _ObservationDatePatch:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._date: ContextVar[str | None] = ContextVar("mem0_observation_date", default=None)
        self.wrapper: Callable[..., Any] | None = None

    @contextmanager
    def active(self, created_at: str) -> Iterator[None]:
        self.install()
        token = self._date.set(created_at)
        try:
            yield
        finally:
            self._date.reset(token)

    def install(self) -> None:
        mem0_main = importlib.import_module("mem0.memory.main")
        with self._lock:
            if mem0_main.generate_additive_extraction_prompt is self.wrapper:
                return
            original = mem0_main.generate_additive_extraction_prompt

            def with_observation_date(*args: Any, **kwargs: Any) -> Any:
                created_at = self._date.get()
                if created_at is not None:
                    kwargs.setdefault("timestamp", created_at)
                return original(*args, **kwargs)

            self.wrapper = with_observation_date
            mem0_main.generate_additive_extraction_prompt = with_observation_date


OBSERVATION_DATE = _ObservationDatePatch()


def mem0_observation_date(created_at: str) -> Any:
    """Within the block, Mem0 extraction prompts carry ``created_at`` as their observation date."""
    return OBSERVATION_DATE.active(created_at)
