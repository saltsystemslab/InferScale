from __future__ import annotations

from typing import Any

FILTER_OPERATORS = frozenset({"eq", "in", "ne"})


def payload_matches(payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = payload.get(key)
        if isinstance(expected, dict):
            if not FILTER_OPERATORS.issuperset(expected):
                raise _unsupported_operators(key, expected)
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


def _unsupported_operators(key: str, expected: dict[str, Any]) -> ValueError:
    unsupported = sorted(repr(operator) for operator in set(expected) - FILTER_OPERATORS)
    supported = sorted(repr(operator) for operator in FILTER_OPERATORS)
    return ValueError(
        f"Filter on {key!r} uses unsupported operators {', '.join(unsupported)}; "
        f"supported operators are {', '.join(supported)}."
    )
