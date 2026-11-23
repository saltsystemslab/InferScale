from __future__ import annotations

import pytest

from inferscale.v1.index.matmul import MatmulIndex


def test_added_vectors_wait_for_the_next_finalize_before_they_have_stable_ids() -> None:
    index = MatmulIndex()
    index.finalize()
    assert index.stable_id_items() == ()

    index.add_many([[1.0, 0.0], [0.0, 1.0]], [{"row": 0}, {"row": 1}], ["a", "b"])

    with pytest.raises(RuntimeError) as unavailable:
        index.stable_id_items()
    assert str(unavailable.value) == "Exact search stable IDs are unavailable before finalization."
    assert (index.vector_count, index.dim) == (2, 2)
