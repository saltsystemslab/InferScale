from __future__ import annotations

from typing import Any


def payload_matches(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = payload.get(key)
        if isinstance(expected, dict):
            if "eq" in expected and actual != expected["eq"]:
                return False
            if "in" in expected and actual not in expected["in"]:
                return False
            if "ne" in expected and actual == expected["ne"]:
                return False
            continue
        if actual != expected:
            return False
    return True
