from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any


def load_corpus(
    catalog_dir: Path, *, seed: int = 0, max_facts: int | None = None
) -> tuple[list[str], dict[str, Any]]:
    """Read one catalog leaf directory, preserving exact text and multiplicity."""
    if max_facts is not None and max_facts < 1:
        raise ValueError("max_facts must be positive")
    paths = sorted(catalog_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No catalogs in {catalog_dir}; select a leaf containing sample JSON files")
    texts: list[str] = []
    sources = []
    samples: set[str] = set()
    identity = None
    for path in paths:
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict) or payload.get("version") != 5:
            raise ValueError(f"Expected a version 5 Mem0 fact catalog: {path}")
        sample = payload.get("sample_id")
        if not isinstance(sample, str) or not sample or sample in samples:
            raise ValueError(f"Missing or duplicate sample_id in {path}")
        samples.add(sample)
        current = {k: v for k, v in payload.items()
                   if k not in {"facts", "sample_id", "sample_fingerprint"}}
        if identity is not None and current != identity:
            raise ValueError(f"Mixed extraction/embedding catalog identities in {path}")
        identity = current
        facts = payload.get("facts")
        if not isinstance(facts, list):
            raise ValueError(f"Missing facts list in {path}")
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
        sources.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
                        "sample_id": sample, "fact_count": len(facts)})
    available = len(texts)
    if not available:
        raise ValueError("The selected catalogs contain no facts")
    random.Random(seed).shuffle(texts)
    if max_facts is not None:
        texts = texts[:max_facts]
    digest = hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    return texts, {"sources": sources, "catalog_identity": identity, "seed": seed,
                   "available_facts": available, "selected_facts": len(texts),
                   "unique_texts": len(set(texts)), "ordered_texts_sha256": digest}
