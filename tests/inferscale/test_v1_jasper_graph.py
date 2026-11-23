from __future__ import annotations

import pytest

from inferscale.v1.index.jasper_graph import serialized_bytes

HEADER_BYTES = 32


@pytest.mark.parametrize(
    "vector_count,dim,n_neighbors,expected_bytes",
    [
        (1, 1, 0, HEADER_BYTES + 2 + 1),
        (1000, 1536, 64, HEADER_BYTES + 1000 * (1536 * 2 + 1 + 64 * 4)),
    ],
)
def test_serialized_graph_is_a_header_and_per_vector_its_float16_values_edge_count_and_edges(
    vector_count, dim, n_neighbors, expected_bytes
) -> None:
    assert (
        serialized_bytes(vector_count=vector_count, dim=dim, n_neighbors=n_neighbors)
        == expected_bytes
    )
