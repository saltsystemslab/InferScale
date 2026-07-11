from __future__ import annotations

import json
from pathlib import Path

from locomo_jasper_bench import cache_identity
from locomo_jasper_bench.cache_identity import atomic_write_json, safe_path_part
from locomo_jasper_bench.embedding import cache as embedding_cache
from locomo_jasper_bench.retrieval import fact_catalog, memory_llm_cache


def test_all_stores_share_the_same_path_sanitizer() -> None:
    assert embedding_cache.safe_path_part is cache_identity.safe_path_part
    assert fact_catalog.safe_path_part is cache_identity.safe_path_part
    assert memory_llm_cache.safe_path_part is cache_identity.safe_path_part


def test_safe_path_part_behavior() -> None:
    assert safe_path_part("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama_Llama-3.1-8B-Instruct"
    assert safe_path_part("a//b") == "a_b"
    assert safe_path_part("a b") == "a_b"
    assert safe_path_part("  ") == "default"


def test_atomic_write_json_compact_and_indented(tmp_path: Path) -> None:
    compact = tmp_path / "compact.json"
    atomic_write_json(compact, {"b": 1, "a": 2})
    compact_text = compact.read_text(encoding="utf-8")
    assert compact_text == '{"a":2,"b":1}\n'

    indented = tmp_path / "nested" / "indented.json"
    atomic_write_json(indented, {"b": 1, "a": 2}, indent=2)
    indented_text = indented.read_text(encoding="utf-8")
    assert indented_text.startswith('{\n  "a": 2')
    assert json.loads(indented_text) == {"a": 2, "b": 1}

    assert not list(tmp_path.glob("*.tmp"))
