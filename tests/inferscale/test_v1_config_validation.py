from __future__ import annotations

import copy
from typing import Any

from inferscale.v1 import MAX_JASPER_BEAM_WIDTH, InferScaleConfig

JASPER_CONFIG = {"model": "m", "vector_backend": "jasper"}


def with_settings(config: dict[str, Any], *settings: dict[str, Any]) -> dict[str, Any]:
    changed = copy.deepcopy(config)
    for setting in settings:
        for key, value in setting.items():
            if isinstance(value, dict):
                changed[key] = {**changed.get(key, {}), **value}
            else:
                changed[key] = value
    return changed


def test_widest_effective_beam_jasperpy_accepts_is_959() -> None:
    config = InferScaleConfig.from_dict(
        with_settings(JASPER_CONFIG, {"top_k": 959, "index": {"beam_width": 959}})
    )

    assert MAX_JASPER_BEAM_WIDTH == 959
    assert (config.top_k, config.index.beam_width) == (959, 959)
