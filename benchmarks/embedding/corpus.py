"""The fact texts of one Mem0 fact catalog directory, as the embedding benchmark's corpus."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

FACT_CATALOG_VERSION = 5
# Catalog keys that vary per sample; every other key must agree across the directory.
PER_SAMPLE_KEYS = frozenset({"facts", "sample_id", "sample_fingerprint"})


def load_corpus(
    catalog_dir: Path, *, seed: int = 0, max_facts: int | None = None
) -> tuple[list[str], dict[str, Any]]:
    """Read one catalog leaf directory, preserving exact text and multiplicity."""
    if max_facts is not None and max_facts < 1:
        raise ValueError("max_facts must be positive")
    paths = sorted(catalog_dir.glob("*.json"))
    if not paths:
        raise ValueError(
            f"No catalogs in {catalog_dir}; select a leaf containing sample JSON files"
        )
    texts: list[str] = []
    sources: list[dict[str, Any]] = []
    samples: set[str] = set()
    identity: dict[str, Any] | None = None
    for path in paths:
        raw = path.read_bytes()
        payload = _catalog_payload(path, raw)
        sample = _sample_id(path, payload, samples)
        current = {key: value for key, value in payload.items() if key not in PER_SAMPLE_KEYS}
        if identity is not None and current != identity:
            raise ValueError(f"Mixed extraction/embedding catalog identities in {path}")
        identity = current
        facts = _fact_texts(path, payload, sample)
        texts.extend(facts)
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "sample_id": sample,
                "fact_count": len(facts),
            }
        )
    available = len(texts)
    if not available:
        raise ValueError("The selected catalogs contain no facts")
    random.Random(seed).shuffle(texts)
    if max_facts is not None:
        texts = texts[:max_facts]
    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    return texts, {
        "sources": sources,
        "catalog_identity": identity,
        "seed": seed,
        "available_facts": available,
        "selected_facts": len(texts),
        "unique_texts": len(set(texts)),
        "ordered_texts_sha256": digest,
    }


def _catalog_payload(path: Path, raw: bytes) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("version") != FACT_CATALOG_VERSION:
        raise ValueError(f"Expected a version {FACT_CATALOG_VERSION} Mem0 fact catalog: {path}")
    return payload


def _sample_id(path: Path, payload: dict[str, Any], seen: set[str]) -> str:
    sample = payload.get("sample_id")
    if not isinstance(sample, str) or not sample or sample in seen:
        raise ValueError(f"Missing or duplicate sample_id in {path}")
    seen.add(sample)
    return sample


def _fact_texts(path: Path, payload: dict[str, Any], sample: str) -> list[str]:
    facts = payload.get("facts")
    if not isinstance(facts, list):
        raise ValueError(f"Missing facts list in {path}")
    texts: list[str] = []
    ids: set[str] = set()
    for fact in facts:
        if not isinstance(fact, dict):
            raise ValueError(f"Invalid fact in {path}")
        fact_id, text = fact.get("id"), fact.get("text")
        if not isinstance(fact_id, str) or not fact_id or fact_id in ids:
            raise ValueError(f"Missing or duplicate fact id in {path}")
        if fact.get("sample_id") != sample or not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid fact text or sample_id in {path}")
        ids.add(fact_id)
        texts.append(text)
    return texts
