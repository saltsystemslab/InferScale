from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from benchmarks.embedding.corpus import load_corpus


def test_manifest_fingerprints_every_catalog_and_the_selected_texts(
    tmp_path: Path, write_catalog: Any
) -> None:
    first = write_catalog(tmp_path / "a.json", "sample-a", ["café", "plain", "plain"])
    second = write_catalog(tmp_path / "b.json", "sample-b", ["日本語"])

    texts, manifest = load_corpus(tmp_path, seed=7)

    assert list(manifest.items()) == [
        (
            "sources",
            [
                {
                    "path": str(first.resolve()),
                    "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
                    "sample_id": "sample-a",
                    "fact_count": 3,
                },
                {
                    "path": str(second.resolve()),
                    "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
                    "sample_id": "sample-b",
                    "fact_count": 1,
                },
            ],
        ),
        (
            "catalog_identity",
            {"version": 6, "model": "extractor", "embedding_model": "Qwen/Qwen3-Embedding-0.6B"},
        ),
        ("seed", 7),
        ("available_facts", 4),
        ("selected_facts", 4),
        ("unique_texts", 3),
        (
            "ordered_texts_sha256",
            hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        ),
    ]
    assert (
        manifest["ordered_texts_sha256"] != hashlib.sha256(json.dumps(texts).encode()).hexdigest()
    )


@pytest.mark.parametrize("seed", [12345])
def test_facts_are_shuffled_by_the_seed_from_their_order_in_the_sorted_catalogs(
    tmp_path: Path, write_catalog: Any, seed: int
) -> None:
    write_catalog(tmp_path / "b.json", "sample-b", [f"b{index}" for index in range(5)])
    write_catalog(tmp_path / "a.json", "sample-a", [f"a{index}" for index in range(6)])
    in_catalog_order = [f"a{index}" for index in range(6)] + [f"b{index}" for index in range(5)]
    shuffled = list(in_catalog_order)
    random.Random(seed).shuffle(shuffled)

    texts, manifest = load_corpus(tmp_path, seed=seed)
    subset, subset_manifest = load_corpus(tmp_path, seed=seed, max_facts=4)

    assert texts == shuffled
    assert subset == shuffled[:4]
    assert (manifest["seed"], manifest["available_facts"], manifest["selected_facts"]) == (
        seed,
        11,
        11,
    )
    assert (subset_manifest["available_facts"], subset_manifest["selected_facts"]) == (11, 4)
    assert load_corpus(tmp_path, seed=seed, max_facts=50)[0] == shuffled


def test_catalogs_may_differ_only_in_their_facts_and_sample_keys(
    tmp_path: Path, write_catalog: Any
) -> None:
    write_catalog(tmp_path / "a.json", "sample-a", ["a"], sample_fingerprint="one")
    write_catalog(tmp_path / "b.json", "sample-b", ["b"], sample_fingerprint="two")

    _, manifest = load_corpus(tmp_path)

    assert manifest["catalog_identity"] == {
        "version": 6,
        "model": "extractor",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
    }

    write_catalog(tmp_path / "c.json", "sample-c", ["c"], extraction_prompt="another")
    with pytest.raises(
        ValueError, match="Mixed extraction/embedding catalog identities in .*c.json"
    ):
        load_corpus(tmp_path)
