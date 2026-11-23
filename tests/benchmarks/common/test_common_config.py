from __future__ import annotations

import re
from typing import Any

import pytest

from benchmarks.common.inferscale_config import inferscale_section
from benchmarks.common.json_fields import ConfigError
from inferscale.v1 import InferScaleConfig


def library_config(section: dict[str, Any]) -> InferScaleConfig:
    return inferscale_section(
        {"inferscale": section}, model="org/model", top_k=10, context_window=0, where="memory"
    )


def test_library_config_still_validates_jasper_tuning_although_benchmarks_pick_backends_elsewhere():
    assert library_config({}).vector_backend == "jasper"
    with pytest.raises(ConfigError) as too_wide:
        library_config({"index": {"beam_width": 100_000}})
    assert re.fullmatch(
        r"memory\.inferscale: Effective Jasper beam width must be <= \d+; got max\(100000, 10\)\.",
        str(too_wide.value),
    )
    with pytest.raises(
        ConfigError, match=r"^memory\.inferscale: index\.n_neighbors must be >= 1\.$"
    ):
        library_config({"index": {"n_neighbors": 0}})


def test_mistyped_library_setting_is_reported_with_its_location_once():
    with pytest.raises(ConfigError) as rejected:
        library_config({"engine": {"block_size": "16"}})

    assert str(rejected.value) == (
        "memory.inferscale.engine.block_size must be an integer, got str."
    )
