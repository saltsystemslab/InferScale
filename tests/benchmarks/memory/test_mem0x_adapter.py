from __future__ import annotations

import pytest

from benchmarks.common.vector_types import SearchHit, SearchMetrics
from benchmarks.memory.mem0.adapter import Mem0JasperVectorStore
from benchmarks.memory.mem0.fact_catalog import FACT_METADATA_KEYS


@pytest.fixture
def jasper_store(tmp_path):
    return Mem0JasperVectorStore(path=tmp_path, backend="jasper")


def stored_payload(vector_store, payload, requested_id="stored"):
    [stored_id] = vector_store.insert(vectors=[[1.0, 0.0]], payloads=[payload], ids=[requested_id])
    return vector_store.get(stored_id).payload


def test_ids_must_pair_with_the_payloads_one_to_one(jasper_store):
    with pytest.raises(ValueError, match="ids and payloads must have the same length"):
        jasper_store.insert(
            vectors=[[1.0, 0.0]], payloads=[{"data": "one"}], ids=["one", "one-too-many"]
        )

    assert jasper_store.col_info()["vectors"] == 0


def test_empty_id_is_refused_instead_of_naming_a_vector(jasper_store):
    with pytest.raises(ValueError, match="Mem0 vector ids must not be empty"):
        jasper_store.insert(vectors=[[1.0, 0.0]], payloads=[{"fact_id": ""}], ids=["mem0-1"])

    assert jasper_store.col_info()["vectors"] == 0


def test_every_key_stored_with_a_fact_is_mirrored_in_both_directions(jasper_store):
    only_in_metadata = {key: f"{key} from the metadata" for key in FACT_METADATA_KEYS}
    only_at_top_level = {key: f"{key} from the top level" for key in FACT_METADATA_KEYS}

    promoted = stored_payload(jasper_store, {"data": "x", "metadata": dict(only_in_metadata)})
    nested = stored_payload(jasper_store, {"data": "x", **only_at_top_level})

    assert {key: promoted[key] for key in FACT_METADATA_KEYS} == only_in_metadata
    assert nested["metadata"] == only_at_top_level


class ShortStore:
    def __init__(self, hits, vector_count):
        self.hits = hits
        self.vector_count = vector_count
        self.searches = []

    def search(self, vector, top_k, filters):
        self.searches.append((vector.tolist(), top_k, filters))
        return self.hits, SearchMetrics(7.0, vector_backend="jasper")

    def count(self, filters):
        return self.vector_count


def test_search_refuses_fewer_hits_than_the_store_holds(jasper_store):
    only_hit = SearchHit(id="east", payload={}, score=0.9, distance=-0.9, rank=1)
    jasper_store.store = ShortStore([only_hit], vector_count=3)
    metrics_before = jasper_store.last_search_metrics

    with pytest.raises(RuntimeError, match="jasper returned 1 hits, expected 2"):
        jasper_store.search(query="ignored", vectors=[0.5, 0.25], top_k=2, filters={"user_id": "u"})

    assert jasper_store.store.searches == [([0.5, 0.25], 2, {"user_id": "u"})]
    assert jasper_store.last_search_metrics is metrics_before
    assert jasper_store.search(query="ignored", vectors=[0.5, 0.25], top_k=1) == [only_hit]
    assert jasper_store.last_search_metrics.search_time_ms == 7.0
