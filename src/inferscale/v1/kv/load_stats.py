from __future__ import annotations

import threading


class LoadStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests_loaded = 0
        self._tokens_loaded = 0
        self._loaded_request_ids: set[str] = set()
        self._native_covered_request_ids: set[str] = set()

    def record_load(self, num_tokens: int, request_id: str = "") -> None:
        with self._lock:
            self._requests_loaded += 1
            self._tokens_loaded += int(num_tokens)
            if request_id:
                self._loaded_request_ids.add(str(request_id))

    def record_native_covered(self, request_id: str) -> None:
        with self._lock:
            if request_id:
                self._native_covered_request_ids.add(str(request_id))

    def reset(self) -> None:
        with self._lock:
            self._requests_loaded = 0
            self._tokens_loaded = 0
            self._loaded_request_ids.clear()
            self._native_covered_request_ids.clear()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            covered = self._loaded_request_ids | self._native_covered_request_ids
            return {
                "requests_loaded": self._requests_loaded,
                "tokens_loaded": self._tokens_loaded,
                "requests_covered": len(covered),
            }
