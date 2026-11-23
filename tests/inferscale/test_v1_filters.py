from __future__ import annotations

import pytest

from inferscale.v1.index.filters import payload_matches


def test_filter_with_an_unsupported_operator_is_rejected_instead_of_matching() -> None:
    with pytest.raises(ValueError, match=r"'n' uses unsupported operators 'gt', 'lt'"):
        payload_matches({"n": 2}, {"n": {"lt": 5, "eq": 2, "gt": 1}})

    assert payload_matches({"n": 2}, {"n": {"eq": 2, "ne": 3, "in": [1, 2]}})
    assert payload_matches({"n": 2}, {"n": {}})
