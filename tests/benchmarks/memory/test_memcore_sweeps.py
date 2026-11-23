from __future__ import annotations

from benchmarks.memory.sweeps import MemoryCell


def test_cell_run_id_names_label_variant_backend_samples_top_k_window_and_stamp():
    cell = MemoryCell("kv", "kv-injection", "jasper", top_k=50, context_window=20)

    assert cell.run_id("llama", 10, "STAMP") == "llama-kv-mem0-jasper10-k50-s20-STAMP"
    assert cell.run_id("llama", None, "STAMP") == "llama-kv-mem0-jasperall-k50-s20-STAMP"
    assert cell.run_id("llama", 0, "STAMP") == "llama-kv-mem0-jasper0-k50-s20-STAMP"
