from __future__ import annotations

import pytest

from benchmarks.common.vector_types import SearchHit
from benchmarks.memory.mem0.search_contract import validate_search_hits


def hit(item_id, rank, *, distance=0.5, score=-0.5):
    return SearchHit(id=item_id, payload={}, score=score, distance=distance, rank=rank)


@pytest.mark.parametrize("distance,score", [(1e30, 0.0), (0.0, -1e30)])
def test_distance_or_score_as_large_as_an_unfilled_jasper_slot_is_rejected(distance, score):
    hits = [hit("one", 1), hit("two", 2, distance=distance, score=score)]

    with pytest.raises(RuntimeError) as error:
        validate_search_hits(hits, expected_count=2, backend="jasper")

    assert str(error.value) == (
        "jasper returned a sentinel-like score or distance for result 'two'."
    )
