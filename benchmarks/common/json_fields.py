"""Validated access to the JSON configuration files every benchmark reads.

Benchmarks are configured only by JSON files under ``configs/``. These helpers
turn missing keys, wrong types, and unknown keys into ``ConfigError`` with the
JSON path in the message, so a mistake in a config file fails before any model
or server is started.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_type_hints


class ConfigError(ValueError):
    """A configuration file is missing, malformed, or holds an invalid value."""


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Read a JSON file whose top level must be an object."""
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Config file is not valid JSON: {target} ({exc})") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config file must contain a JSON object: {target}")
    return raw


def require(
    section: Mapping[str, Any], key: str, expected: type | tuple[type, ...], where: str
) -> Any:
    """Return ``section[key]`` after checking its type; the key must be present."""
    if key not in section:
        raise ConfigError(f"{where} is missing required key {key!r}.")
    return typed_value(section[key], key, expected, where)


def optional(
    section: Mapping[str, Any],
    key: str,
    expected: type | tuple[type, ...],
    default: Any,
    where: str,
) -> Any:
    """Return ``section[key]`` after checking its type, or ``default`` when absent or null."""
    if key not in section or section[key] is None:
        return default
    return typed_value(section[key], key, expected, where)


def reject_unknown_keys(section: Mapping[str, Any], allowed: Sequence[str], where: str) -> None:
    unknown = sorted(set(section) - set(allowed))
    if unknown:
        raise ConfigError(f"{where} has unknown keys: {', '.join(unknown)}.")


def section_of(
    data: Mapping[str, Any], key: str, where: str, *, required: bool = True
) -> dict[str, Any]:
    """Return the nested object at ``key``; a missing optional section is an empty dict."""
    value = data.get(key)
    if value is None:
        if required:
            raise ConfigError(f"{where} is missing required section {key!r}.")
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{where}.{key} must be a JSON object.")
    return value


def string_map(section: Mapping[str, Any], where: str) -> dict[str, str]:
    """An object whose values are all strings, such as a model alias table."""
    result: dict[str, str] = {}
    for key, value in section.items():
        if not isinstance(value, str):
            raise ConfigError(f"{where}.{key} must be a string.")
        result[str(key)] = value
    return result


def int_list(
    section: Mapping[str, Any], key: str, where: str, *, default: Sequence[int] | None = None
) -> tuple[int, ...]:
    return _unique_list(section, key, where, item_type=int, default=default)


def str_list(
    section: Mapping[str, Any], key: str, where: str, *, default: Sequence[str] | None = None
) -> tuple[str, ...]:
    return _unique_list(section, key, where, item_type=str, default=default)


def _unique_list(
    section: Mapping[str, Any],
    key: str,
    where: str,
    *,
    item_type: type,
    default: Sequence[Any] | None,
) -> tuple[Any, ...]:
    if key not in section or section[key] is None:
        if default is None:
            raise ConfigError(f"{where} is missing required key {key!r}.")
        return tuple(default)
    values = section[key]
    well_typed = isinstance(values, list) and all(
        isinstance(item, item_type) and not isinstance(item, bool) for item in values
    )
    if not well_typed or not values:
        noun = "integers" if item_type is int else "strings"
        raise ConfigError(f"{where}.{key} must be a non-empty list of {noun}.")
    if len(set(values)) != len(values):
        raise ConfigError(f"{where}.{key} must not contain duplicates.")
    return tuple(values)


def expand_path(value: str | Path, *, root: Path) -> Path:
    """Expand ``~`` and environment variables; relative paths are taken from ``root``."""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return expanded if expanded.is_absolute() else root / expanded


def parse_dataclass_section(section: Mapping[str, Any], schema: type, where: str) -> Any:
    """Build a flat dataclass from a JSON object, taking defaults from the dataclass.

    Field names are the allowed keys and the annotations are the accepted JSON
    types, so adding a field to the dataclass is the whole change.
    """
    hints = get_type_hints(schema)
    reject_unknown_keys(section, tuple(hints), where)
    values: dict[str, Any] = {}
    for field in fields(schema):
        if section.get(field.name) is None:
            continue
        values[field.name] = typed_value(
            section[field.name], field.name, accepted_types(hints[field.name]), where
        )
    return schema(**values)


def validate_dataclass_types(data: Mapping[str, Any], schema: type, where: str) -> None:
    """Reject JSON type mismatches for a nested dataclass tree before it is constructed."""
    hints = get_type_hints(schema)
    reject_unknown_keys(data, tuple(hints), where)
    for name, value in data.items():
        expected = hints[name]
        if value is None and type(None) in get_args(expected):
            continue
        if is_dataclass(expected):
            if not isinstance(value, dict):
                raise ConfigError(f"{where}.{name} must be a JSON object.")
            validate_dataclass_types(value, expected, f"{where}.{name}")
            continue
        typed_value(value, name, accepted_types(expected), where)


def accepted_types(hint: Any) -> tuple[type, ...]:
    """The runtime types a JSON value may have for a type hint such as ``float | None``."""
    choices = tuple(choice for choice in (get_args(hint) or (hint,)) if choice is not type(None))
    if float in choices and int not in choices:
        choices = (*choices, int)
    return choices


def typed_value(value: Any, key: str, expected: type | tuple[type, ...], where: str) -> Any:
    """Return ``value`` when it is an instance of ``expected``; booleans never pass as integers."""
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    if bool not in expected_types and isinstance(value, bool):
        raise ConfigError(f"{where}.{key} must be {_describe(expected_types)}, got a boolean.")
    if not isinstance(value, expected_types):
        raise ConfigError(
            f"{where}.{key} must be {_describe(expected_types)}, got {type(value).__name__}."
        )
    return value


def _describe(expected_types: tuple[type, ...]) -> str:
    names = {
        int: "an integer",
        float: "a number",
        str: "a string",
        bool: "a boolean",
        list: "a list",
        dict: "an object",
    }
    return " or ".join(names.get(item, item.__name__) for item in expected_types)
