from __future__ import annotations

import builtins
from typing import Any

import pytest

from inferscale.v1.kv import chunk_map
from inferscale.v1.kv.chunk_map import GPUChunkMap


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:-1", "cuda:abc", "", None])
def test_non_cuda_target_is_rejected_before_torch_import(
    monkeypatch: pytest.MonkeyPatch, device: Any
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch":
            pytest.fail("Invalid device attempted to import torch.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        GPUChunkMap(["fact-a"], device=device)


@pytest.mark.parametrize("ids", [["fact-a", "fact-a"], ["", ""]])
def test_duplicate_registration_fails_before_device_allocation(ids: list[str]) -> None:
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        GPUChunkMap(ids, device="cuda:0")


@pytest.mark.parametrize("ids", ["fact-a", b"fact-a", ["fact-a", 2], [None]])
def test_registration_requires_string_ids(ids: Any) -> None:
    with pytest.raises(TypeError, match="string"):
        GPUChunkMap(ids, device="cuda:0")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("fact-a", (-8007207963077507213, 725902087184088496)),
        ("é", (5375421630974772051, -7069829644800747574)),
        ("", (-2039914840885289964, -7278955230309402332)),
    ],
)
def test_fingerprints_are_deterministic_signed_int64_pairs(
    text: str, expected: tuple[int, int]
) -> None:
    assert chunk_map._fingerprint(text) == expected


@pytest.mark.parametrize("secondary", [11, 12])
def test_primary_hash_collision_fails_closed(
    monkeypatch: pytest.MonkeyPatch, secondary: int
) -> None:
    monkeypatch.setattr(
        chunk_map, "_fingerprint", lambda value: (7, 11 if value == "a" else secondary)
    )
    with pytest.raises(ValueError, match="hash collision"):
        GPUChunkMap(["a", "b"], device="cuda:0")


@pytest.mark.parametrize(
    ("bindings", "error"),
    [
        ([], "without stable-ID bindings"),
        ([(True, "a")], "integers"),
        ([(0.0, "a")], "integers"),
        ([(-1, "a")], "non-negative"),
        ([(0, "a"), (0, "b")], "duplicates"),
        ([(1, "a")], "contiguous"),
        ([(0, "a"), (2, "b")], "contiguous"),
        ([(0, "a"), (1, "a")], "exactly one"),
    ],
)
def test_stable_binding_validation(bindings: Any, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        chunk_map._stable_binding_ids(bindings)


def test_stable_bindings_are_ordered_by_stable_id() -> None:
    assert chunk_map._stable_binding_ids(iter([(2, "b"), (0, "c"), (1, "a")])) == [
        "c", "a", "b"
    ]


@pytest.fixture
def cuda() -> Any:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise resident chunk lookup.")
    return torch


def test_cuda_lookup_preserves_registration_and_query_order(cuda: Any) -> None:
    mapping = GPUChunkMap(["fact-z", "fact-a", "fact-é", ""], device="cuda:0")
    try:
        result = mapping.lookup(["fact-é", "fact-z", "fact-a", "fact-é", ""])
        assert result.device.type == "cuda"
        assert result.dtype == cuda.long
        assert result.tolist() == [2, 0, 1, 2, 3]
        assert mapping.nbytes == 4 * 3 * 8
        assert mapping.device == result.device
        empty = mapping.lookup([])
        assert empty.device == result.device
        assert empty.dtype == cuda.long
        assert empty.numel() == 0
    finally:
        mapping.close()
    assert mapping.nbytes == 0
    with pytest.raises(RuntimeError, match="closed"):
        mapping.lookup(["fact-a"])


def test_cuda_lookup_matches_large_catalog(cuda: Any) -> None:
    ids = [f"fact-{row:05d}" for row in range(4096)]
    mapping = GPUChunkMap(ids, device="cuda")
    try:
        expected_rows = list(range(4095, -1, -17))
        actual = mapping.lookup([ids[row] for row in expected_rows])
        assert actual.tolist() == expected_rows
    finally:
        mapping.close()


def test_cuda_lookup_rejects_missing_and_empty_catalog(cuda: Any) -> None:
    for catalog in ([], ["fact-a", "fact-b"]):
        mapping = GPUChunkMap(catalog, device="cuda:0")
        try:
            assert mapping.lookup([]).numel() == 0
            with pytest.raises(ValueError, match="no pre-encoded KV chunk"):
                mapping.lookup(["missing"])
        finally:
            mapping.close()


def test_cuda_lookup_checks_secondary_hash_for_unknown_id(
    cuda: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        chunk_map, "_fingerprint", lambda value: (7, 11 if value == "stored" else 12)
    )
    mapping = GPUChunkMap(["stored"], device="cuda:0")
    try:
        assert mapping.lookup(["stored"]).tolist() == [0]
        with pytest.raises(ValueError, match="no pre-encoded KV chunk"):
            mapping.lookup(["unknown"])
    finally:
        mapping.close()


def test_cuda_stable_map_maps_to_registration_rows(cuda: Any) -> None:
    mapping = GPUChunkMap(["fact-a", "fact-b", "fact-c"], device="cuda:0")
    try:
        rows = mapping.stable_id_row_map([(1, "fact-a"), (0, "fact-c")])
        assert rows.device.type == "cuda"
        assert rows.dtype == cuda.long
        assert rows.tolist() == [2, 0]
        with pytest.raises(ValueError, match="no pre-encoded KV chunk"):
            mapping.stable_id_row_map([(0, "missing")])
    finally:
        mapping.close()
