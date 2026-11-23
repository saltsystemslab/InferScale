from __future__ import annotations

import numpy as np
import pytest

from inferscale.v1.index.jasper import JasperIndex

VECTORS = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]
PAYLOADS = [{"group": "keep"}, {"group": "drop"}, {"group": "keep"}, {"group": "keep"}]
IDS = ["a", "b", "c", "d"]
QUERY = [1.0, 2.0, -1.0]
IDS_BY_SCORE = ["b", "c", "a", "d"]


class ResidentGraph:
    def __init__(self) -> None:
        self.times_freed = 0

    def free(self) -> None:
        self.times_freed += 1


@pytest.fixture
def index() -> JasperIndex:
    index = JasperIndex()
    index.add_many(VECTORS, PAYLOADS, IDS)
    return index


@pytest.fixture
def graph(index, monkeypatch) -> ResidentGraph:
    graph = ResidentGraph()
    monkeypatch.setattr(index, "_build_jasper_graph", lambda: graph)
    index.finalize()
    return graph


def ids_of(hits) -> list[str]:
    return [hit.id for hit in hits]


def record_beam_searches(index: JasperIndex, monkeypatch) -> list[tuple[int, int]]:
    searches: list[tuple[int, int]] = []

    def beam_search(query, top_k, *, beam_width):
        searches.append((top_k, beam_width))
        return index._search_exact(query, top_k)

    monkeypatch.setattr(index, "_search_jasper", beam_search)
    return searches


def test_batches_are_appended_in_order_under_the_given_ids() -> None:
    index = JasperIndex()

    first = index.add_many(VECTORS[:2], PAYLOADS[:2], IDS[:2])
    second = index.add_many(
        (np.asarray(vector) for vector in VECTORS[2:]),
        (dict(payload) for payload in PAYLOADS[2:]),
        (number for number in (3, 4)),
    )

    assert (first, second) == (["a", "b"], ["3", "4"])
    assert (index.vector_count, index.dim) == (4, 3)
    assert index.rows() == list(zip(["a", "b", "3", "4"], PAYLOADS))
    assert ids_of(index.search(QUERY, top_k=4)[0]) == ["b", "3", "a", "4"]


def test_added_vectors_free_the_graph_built_for_the_previous_matrix(index, graph) -> None:
    assert index.memory_stats()["jasper_embedding_matrix_gpu_logical_bytes"] == 48

    index.add_many([[9.0, 9.0, 9.0]], [{"group": "late"}], ["late"])

    assert graph.times_freed == 1
    assert index.memory_stats()["jasper_embedding_matrix_gpu_logical_bytes"] == 0
    with pytest.raises(RuntimeError, match="unavailable before finalization"):
        index.stable_id_items()
    assert ids_of(index.search(QUERY, top_k=1)[0]) == ["late"]


def test_memory_stats_describe_the_host_matrix_before_a_graph_exists(index) -> None:
    assert index.memory_stats() == {
        "jasper_vector_count": 4,
        "jasper_embedding_dim": 3,
        "jasper_embedding_matrix_cpu_bytes": 48,
        "jasper_embedding_matrix_cpu_mb": 48 / (1024 * 1024),
        "jasper_embedding_matrix_gpu_logical_bytes": 0,
        "jasper_embedding_matrix_gpu_logical_mb": 0.0,
        "jasper_graph_gpu_bytes": None,
        "jasper_graph_gpu_mb": None,
        "jasper_graph_torch_allocated_delta_bytes": None,
        "jasper_graph_torch_allocated_delta_mb": None,
    }
    assert list(index.memory_stats())[:2] == ["jasper_vector_count", "jasper_embedding_dim"]


def test_update_replaces_the_payload_or_the_vector_of_one_row(index, graph) -> None:
    payload = {"group": "updated"}

    index.update("d", payload=payload)
    index.update("d", vector=[0.0, 5.0, 0.0])
    payload["group"] = "changed afterwards"

    assert index.rows() == [*zip(IDS[:3], PAYLOADS[:3]), ("d", {"group": "updated"})]
    assert ids_of(index.search(QUERY, top_k=1)[0]) == ["d"]
    assert graph.times_freed == 1
    with pytest.raises(RuntimeError, match="unavailable before finalization"):
        index.stable_id_items()


def test_delete_removes_one_row_and_renumbers_the_ones_after_it(index, graph) -> None:
    index.delete("b")

    assert index.rows() == [("a", PAYLOADS[0]), ("c", PAYLOADS[2]), ("d", PAYLOADS[3])]
    assert [index.get(item_id).rank for item_id in ("a", "c", "d")] == [1, 2, 3]
    assert index.get("b") is None
    assert ids_of(index.search(QUERY, top_k=99)[0]) == ["c", "a", "d"]
    assert graph.times_freed == 1


def test_whole_store_request_is_scored_exactly_because_beam_search_may_skip_vertices(
    index, graph, monkeypatch
) -> None:
    beam_searches = record_beam_searches(index, monkeypatch)

    part_of_the_store, _ = index.search(QUERY, top_k=3)
    whole_store, _ = index.search(QUERY, top_k=4)

    assert beam_searches == [(3, 64)]
    assert ids_of(part_of_the_store) == IDS_BY_SCORE[:3]
    assert ids_of(whole_store) == IDS_BY_SCORE
