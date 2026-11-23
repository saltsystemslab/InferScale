from __future__ import annotations


def bytes_to_mb(byte_count: int | float) -> float:
    return byte_count / (1024 * 1024)
