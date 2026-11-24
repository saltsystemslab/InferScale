from __future__ import annotations

import pytest

from benchmarks.common.config import ConfigError
from benchmarks.memory.sweeps import MemoryCell, parse_sweeps, sweep_cells


def test_cell_run_id_names_label_variant_backend_samples_top_k_window_and_stamp():
    cell = MemoryCell("kv", "kv-injection", "jasper", top_k=50, context_window=20)

    assert cell.run_id("llama", 10, "STAMP") == "llama-kv-mem0-jasper10-k50-s20-STAMP"
    assert cell.run_id("llama", None, "STAMP") == "llama-kv-mem0-jasperall-k50-s20-STAMP"
    assert cell.run_id("llama", 0, "STAMP") == "llama-kv-mem0-jasper0-k50-s20-STAMP"


def test_two_variants_may_share_a_name_only_on_different_vector_backends():
    on_qdrant = {
        "name": "prompt-injection",
        "answer_backend": "prompt-injection",
        "vector_backend": "qdrant",
    }
    on_jasper = {**on_qdrant, "vector_backend": "jasper"}

    sweeps = parse_sweeps(
        {"standard": {"top_k": [5], "variants": [on_qdrant, on_jasper]}}, "sweeps"
    )

    assert [cell.run_id("llama", 10, "STAMP") for cell in sweep_cells(sweeps["standard"])] == [
        "llama-prompt-injection-mem0-qdrant10-k5-s0-STAMP",
        "llama-prompt-injection-mem0-jasper10-k5-s0-STAMP",
    ]
    with pytest.raises(ConfigError, match="run ids would collide"):
        parse_sweeps({"standard": {"top_k": [5], "variants": [on_qdrant, on_qdrant]}}, "sweeps")
