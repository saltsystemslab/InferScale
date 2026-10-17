from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from inferscale.v1.index.matmul import MatmulIndex


VECTORS = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]
PAYLOADS = [{"group": "keep"}, {"group": "drop"}, {"group": "keep"}, {"group": "keep"}]
IDS = ["a", "b", "c", "d"]
QUERY = [1.0, 2.0, -1.0]


def test_import_registration_and_empty_search_need_no_gpu_dependencies():
    code = '''
import importlib.abc
import sys

class BlockGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {"torch", "jasper", "tvm_ffi", "vllm"}:
            raise AssertionError(f"Unexpected GPU dependency import: {fullname}")

sys.meta_path.insert(0, BlockGPUImports())
from inferscale.v1.index.matmul import MatmulIndex

index = MatmulIndex()
hits, metrics = index.search([1.0, 0.0], top_k=4)
assert hits == []
assert metrics.vector_backend == "exact"
assert index.search_device([1.0, 0.0], top_k=4) is None
index.finalize()
assert index.stable_id_items() == ()
assert index.add_many([[1.0, 0.0]], [{"group": "a"}], ["first"]) == ["first"]
assert index.vector_count == 1
index.close()
assert "torch" not in sys.modules
assert "jasper" not in sys.modules
print("MATMUL_IMPORT_OK")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MATMUL_IMPORT_OK" in result.stdout


@pytest.mark.parametrize("vectors,payloads,ids", [
    ([[1.0, 2.0]], [], ["new"]),
    ([[1.0, 2.0]], [{}], []),
    ([[1.0]], [{}], ["new"]),
    ([[]], [{}], ["new"]),
    ([[float("nan"), 1.0]], [{}], ["new"]),
    ([[float("inf"), 1.0]], [{}], ["new"]),
    ([[1.0, 2.0]], [{}], ["existing"]),
    ([[1.0, 2.0], [3.0, 4.0]], [{}, {}], ["same", "same"]),
])
def test_invalid_add_preserves_existing_vectors_and_payloads(vectors, payloads, ids):
    index = MatmulIndex()
    index.add_many([[1.0, 0.0]], [{"saved": True}], ["existing"])

    with pytest.raises(ValueError):
        index.add_many(vectors, payloads, ids)

    assert index.vector_count == 1
    assert index.dim == 2
    assert index.get("existing").payload == {"saved": True}


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:-1", "cuda:abc", "", None])
def test_invalid_device_is_rejected_without_torch(device, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match="CUDA device"):
        MatmulIndex(device=device)


def test_query_dimension_mismatch_fails_before_gpu_import(monkeypatch):
    index = MatmulIndex()
    index.add_many([[1.0, 0.0]], [{}], ["a"])
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ValueError, match="dim"):
        index.search([1.0], top_k=1)
    with pytest.raises(ValueError, match="dim"):
        index.search_device([1.0], top_k=1)


def test_missing_torch_has_a_clear_error(monkeypatch):
    index = MatmulIndex()
    index.add_many([[1.0, 0.0]], [{}], ["a"])
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(RuntimeError, match="(?i)(torch|cuda)"):
        index.finalize()


def test_unavailable_cuda_has_a_clear_error_without_cpu_fallback(monkeypatch):
    index = MatmulIndex()
    index.add_many([[1.0, 0.0]], [{}], ["a"])
    unavailable_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", unavailable_torch)

    with pytest.raises(RuntimeError, match="CUDA"):
        index.finalize()


@pytest.fixture
def cuda_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise GPU matrix multiplication retrieval.")
    return torch


@pytest.fixture
def gpu_index(cuda_torch):
    index = MatmulIndex(device="cuda:0")
    try:
        yield index
    finally:
        index.close()


@pytest.mark.parametrize("count,expected_ids,expected_scores", [
    (1, ["a"], [1.0]),
    (4, ["b", "c", "a", "d"], [4.0, 3.0, 1.0, -2.0]),
])
def test_gpu_ranking_matches_known_inner_products(gpu_index, count, expected_ids, expected_scores):
    gpu_index.add_many(VECTORS[:count], PAYLOADS[:count], IDS[:count])
    gpu_index.finalize()
    hits, metrics = gpu_index.search(QUERY, top_k=99)

    assert [hit.id for hit in hits] == expected_ids
    assert [hit.score for hit in hits] == pytest.approx(expected_scores)
    assert [hit.distance for hit in hits] == pytest.approx([-score for score in expected_scores])
    assert [hit.rank for hit in hits] == list(range(1, count + 1))
    assert metrics.vector_backend == "exact"
    assert metrics.jasper_effective_beam_width is None
    assert metrics.search_time_ms >= 0


def test_device_results_keep_gpu_stable_ids_and_distances(gpu_index, cuda_torch):
    from inferscale.v1.types import DeviceSearchResult

    gpu_index.add_many(VECTORS, PAYLOADS, IDS)
    gpu_index.finalize()
    bindings = list(gpu_index.stable_id_items())
    assert bindings == list(enumerate(IDS))

    result = gpu_index.search_device(QUERY, top_k=3)

    assert isinstance(result, DeviceSearchResult)
    assert result.stable_ids.is_cuda
    assert result.distances.is_cuda
    assert result.stable_ids.dtype == cuda_torch.long
    assert result.distances.dtype == cuda_torch.float32
    assert result.stable_ids.tolist() == [1, 2, 0]
    assert result.distances.tolist() == pytest.approx([-4.0, -3.0, -1.0])
    assert result.metrics.vector_backend == "exact"
    hits = gpu_index.materialize_device_result(result)
    assert [hit.id for hit in hits] == ["b", "c", "a"]
    assert [hit.payload for hit in hits] == [PAYLOADS[1], PAYLOADS[2], PAYLOADS[0]]


@pytest.mark.parametrize("filters,top_k,expected_ids", [
    ({"group": "keep"}, 2, ["c", "a"]),
    ({"group": {"eq": "keep"}}, 99, ["c", "a", "d"]),
    ({"group": {"ne": "keep"}}, 4, ["b"]),
    ({"group": "missing"}, 4, []),
])
def test_gpu_filtering_selects_highest_scoring_matching_rows(gpu_index, filters, top_k, expected_ids):
    gpu_index.add_many(VECTORS, PAYLOADS, IDS)
    hits, metrics = gpu_index.search(QUERY, top_k=top_k, filters=filters)
    assert [hit.id for hit in hits] == expected_ids
    assert [hit.rank for hit in hits] == list(range(1, len(expected_ids) + 1))
    assert metrics.vector_backend == "exact"
    result = gpu_index.search_device(QUERY, top_k=top_k, filters=filters)
    assert result.stable_ids.is_cuda
    assert result.distances.is_cuda
    assert [hit.id for hit in gpu_index.materialize_device_result(result)] == expected_ids


def test_finalize_preserves_fp32_precision_under_ambient_autocast(gpu_index, cuda_torch):
    gpu_index.add_many([[1.0001, 0.0], [1.0002, 0.0]], [{}, {}], ["lower", "higher"])
    with cuda_torch.autocast(device_type="cuda", dtype=cuda_torch.float16):
        hits, _ = gpu_index.search([1.0, 0.0], top_k=1)
    assert hits[0].id == "higher"
    assert hits[0].score == pytest.approx(float(np.float32(1.0002)), abs=1e-7, rel=0)


def test_mutation_invalidates_bindings_and_refinalization_includes_new_rows(gpu_index):
    gpu_index.add_many(VECTORS[:1], PAYLOADS[:1], IDS[:1])
    gpu_index.finalize()
    assert list(gpu_index.stable_id_items()) == [(0, "a")]
    gpu_index.add_many(VECTORS[1:], PAYLOADS[1:], IDS[1:])

    with pytest.raises(RuntimeError):
        list(gpu_index.stable_id_items())

    gpu_index.finalize()
    assert list(gpu_index.stable_id_items()) == list(enumerate(IDS))
    hits, _ = gpu_index.search(QUERY, top_k=1)
    assert [hit.id for hit in hits] == ["b"]

    gpu_index.close()
    with pytest.raises(RuntimeError):
        list(gpu_index.stable_id_items())
    gpu_index.finalize()
    hits, _ = gpu_index.search(QUERY, top_k=1)
    assert [hit.id for hit in hits] == ["b"]
