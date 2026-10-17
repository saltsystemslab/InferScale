from __future__ import annotations

import json
from pathlib import Path

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.cache_keys import endpoint_cache_key, normalize_endpoint, safe_path_part


def test_safe_path_part_behavior() -> None:
    assert safe_path_part("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama_Llama-3.1-8B-Instruct"
    assert safe_path_part("a//b") == "a_b"
    assert safe_path_part("a b") == "a_b"
    assert safe_path_part("  ") == "default"


def test_endpoint_keys_ignore_case_and_trailing_slashes() -> None:
    assert normalize_endpoint("HTTPS://Example.com/v1/") == "https://example.com/v1"
    assert normalize_endpoint(None) == "<provider-default>"
    assert endpoint_cache_key("https://example.com/v1") == endpoint_cache_key(
        "HTTPS://EXAMPLE.COM/v1/"
    )
    assert endpoint_cache_key("https://example.com/v1") != endpoint_cache_key(
        "https://other.example/v1"
    )


def test_atomic_write_json_compact_and_indented(tmp_path: Path) -> None:
    compact = tmp_path / "compact.json"
    atomic_write_json(compact, {"b": 1, "a": 2})
    assert compact.read_text(encoding="utf-8") == '{"a":2,"b":1}\n'

    indented = tmp_path / "nested" / "indented.json"
    atomic_write_json(indented, {"b": 1, "a": 2}, indent=2)
    indented_text = indented.read_text(encoding="utf-8")
    assert indented_text.startswith('{\n  "a": 2')
    assert json.loads(indented_text) == {"a": 2, "b": 1}

    assert not list(tmp_path.glob("*.tmp"))
