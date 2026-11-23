from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from benchmarks.memory.protocol import FACT_CATALOG_FORMAT_VERSION

PER_SAMPLE_KEYS = frozenset({"facts", "sample_id", "sample_fingerprint"})


def load_corpus(
    catalog_dir: Path, *, seed: int = 0, max_facts: int | None = None
) -> tuple[list[str], dict[str, Any]]:
    if max_facts is not None and max_facts < 1:
        raise ValueError("max_facts must be positive")
    catalog_paths = sorted(catalog_dir.glob("*.json"))
    if not catalog_paths:
        raise ValueError(
            f"No catalogs in {catalog_dir}; select a leaf containing sample JSON files"
        )
    all_texts: list[str] = []
    sources: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    directory_identity: dict[str, Any] | None = None
    for catalog_path in catalog_paths:
        catalog_bytes = catalog_path.read_bytes()
        catalog = _parse_catalog(catalog_path, catalog_bytes)
        sample_id = _new_sample_id(catalog_path, catalog, seen_sample_ids)
        seen_sample_ids.add(sample_id)
        identity = _shared_identity(catalog)
        if directory_identity is not None and identity != directory_identity:
            raise ValueError(f"Mixed extraction/embedding catalog identities in {catalog_path}")
        directory_identity = identity
        sample_texts = _fact_texts(catalog_path, catalog, sample_id)
        all_texts.extend(sample_texts)
        sources.append(
            {
                "path": str(catalog_path.resolve()),
                "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
                "sample_id": sample_id,
                "fact_count": len(sample_texts),
            }
        )
    available_facts = len(all_texts)
    if not available_facts:
        raise ValueError("The selected catalogs contain no facts")
    random.Random(seed).shuffle(all_texts)
    selected_texts = all_texts if max_facts is None else all_texts[:max_facts]
    ordered_texts_sha256 = hashlib.sha256(
        json.dumps(selected_texts, ensure_ascii=False).encode()
    ).hexdigest()
    return selected_texts, {
        "sources": sources,
        "catalog_identity": directory_identity,
        "seed": seed,
        "available_facts": available_facts,
        "selected_facts": len(selected_texts),
        "unique_texts": len(set(selected_texts)),
        "ordered_texts_sha256": ordered_texts_sha256,
    }


def _parse_catalog(catalog_path: Path, catalog_bytes: bytes) -> dict[str, Any]:
    catalog = json.loads(catalog_bytes)
    if not isinstance(catalog, dict) or catalog.get("version") != FACT_CATALOG_FORMAT_VERSION:
        raise ValueError(
            f"Expected a version {FACT_CATALOG_FORMAT_VERSION} Mem0 fact catalog: {catalog_path}"
        )
    return catalog


def _new_sample_id(catalog_path: Path, catalog: dict[str, Any], seen_sample_ids: set[str]) -> str:
    sample_id = catalog.get("sample_id")
    if not _is_new_identifier(sample_id, seen_sample_ids):
        raise ValueError(f"Missing or duplicate sample_id in {catalog_path}")
    return sample_id


def _shared_identity(catalog: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in catalog.items() if key not in PER_SAMPLE_KEYS}


def _fact_texts(catalog_path: Path, catalog: dict[str, Any], sample_id: str) -> list[str]:
    facts = catalog.get("facts")
    if not isinstance(facts, list):
        raise ValueError(f"Missing facts list in {catalog_path}")
    texts: list[str] = []
    seen_fact_ids: set[str] = set()
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError(f"Invalid fact in {catalog_path}")
        fact_id, text = fact.get("id"), fact.get("text")
        if not _is_new_identifier(fact_id, seen_fact_ids):
            raise ValueError(f"Missing or duplicate fact id in {catalog_path}")
        if fact.get("sample_id") != sample_id or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid fact text or sample_id in {catalog_path}")
        seen_fact_ids.add(fact_id)
        texts.append(text)
    return texts


def _is_new_identifier(identifier: Any, seen_identifiers: set[str]) -> bool:
    return isinstance(identifier, str) and bool(identifier) and identifier not in seen_identifiers
