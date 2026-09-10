from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from inferscale.v1.kv.chunk_map import GPUChunkMap
from inferscale.v1.kv.prompt import ScaffoldTokens
from inferscale.v1.types import KVChunk
from benchmarks.rag.data_types import RagChunk
from benchmarks.rag.kv_cache import (
    RAG_CHUNK_CACHE_VERSION,
    CpuChunkStore,
    cache_meta_base,
    chunk_file_path,
    chunk_meta,
    chunk_payload_is_valid,
    missing_chunk_files,
    rag_chunk_cache_dir,
    rag_scaffold_chunks_match,
    save_chunk,
    tables_meta,
    tables_payload_is_valid,
)


def _meta_base() -> dict:
    return cache_meta_base(
        dataset="multihoprag",
        model="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
        chunk_size=1024,
        context_window=5,
        max_position=32768,
        corpus_fingerprint="abc123",
    )


def _chunk(doc_id: str = "d1", index: int = 0, token_ids: list[int] | None = None) -> RagChunk:
    return RagChunk(
        chunk_id=f"{doc_id}:{index}",
        doc_id=doc_id,
        chunk_index=index,
        token_ids=token_ids if token_ids is not None else [1, 2, 3, 4],
        text="",
    )


def test_cache_dir_composition(tmp_path) -> None:
    cache_dir = rag_chunk_cache_dir(
        model="meta-llama/Llama-3.1-8B-Instruct",
        dtype="bfloat16",
        chunk_size=1024,
        context_window=5,
        max_position=32768,
        corpus_fingerprint="abc123",
        cache_root=tmp_path,
    )

    assert cache_dir == (
        tmp_path
        / f"v{RAG_CHUNK_CACHE_VERSION}"
        / "meta-llama__Llama-3.1-8B-Instruct"
        / "bfloat16-c1024-w5-p32768"
        / "abc123"
    )


def test_chunk_file_path_sanitizes_ids(tmp_path) -> None:
    assert chunk_file_path(tmp_path, "d1a:0") == tmp_path / "chunks" / "d1a_0.pt"


def test_chunk_payload_validation() -> None:
    meta = chunk_meta(_meta_base(), _chunk())
    payload = {
        "version": RAG_CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "chunk": {"token_ids": [1, 2, 3, 4], "kv_by_layer": {"layer": object()}},
    }

    assert chunk_payload_is_valid(payload, expected_meta=meta, expected_token_ids=[1, 2, 3, 4])
    assert not chunk_payload_is_valid(
        {**payload, "version": 99}, expected_meta=meta, expected_token_ids=[1, 2, 3, 4]
    )
    stale_meta = {**meta, "context_window": 20}
    assert not chunk_payload_is_valid(
        payload, expected_meta=stale_meta, expected_token_ids=[1, 2, 3, 4]
    )
    assert not chunk_payload_is_valid(
        payload, expected_meta=meta, expected_token_ids=[1, 2, 3]
    )
    empty_kv = {**payload, "chunk": {"token_ids": [1, 2, 3, 4], "kv_by_layer": {}}}
    assert not chunk_payload_is_valid(
        empty_kv, expected_meta=meta, expected_token_ids=[1, 2, 3, 4]
    )


def test_tables_payload_validation() -> None:
    meta = tables_meta(_meta_base(), block_size=16)
    payload = {
        "version": RAG_CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "cos_table": object(),
        "sin_table": object(),
        "scaffold_chunks": {"header": {}, "empty_passages": {}, "footer": {}},
    }

    assert tables_payload_is_valid(payload, expected_meta=meta)
    assert not tables_payload_is_valid({**payload, "cos_table": None}, expected_meta=meta)
    missing_slot = {**payload, "scaffold_chunks": {"header": {}, "footer": {}}}
    assert not tables_payload_is_valid(missing_slot, expected_meta=meta)
    assert not tables_payload_is_valid(payload, expected_meta={**meta, "block_size": 32})


def test_scaffold_chunks_match() -> None:
    scaffold = ScaffoldTokens(
        header_token_ids=[1, 2],
        empty_token_ids=[3],
        footer_token_ids=[4, 5],
    )
    chunks = {
        "header": KVChunk(chunk_id="header", token_ids=[1, 2], kv_by_layer={}),
        "empty_passages": KVChunk(chunk_id="empty_passages", token_ids=[3], kv_by_layer={}),
        "footer": KVChunk(chunk_id="footer", token_ids=[4, 5], kv_by_layer={}),
    }

    assert rag_scaffold_chunks_match(chunks, scaffold)
    drifted = dict(chunks)
    drifted["footer"] = KVChunk(chunk_id="footer", token_ids=[4, 6], kv_by_layer={})
    assert not rag_scaffold_chunks_match(drifted, scaffold)
    assert not rag_scaffold_chunks_match({**chunks, "header": None}, scaffold)


def test_missing_chunk_files(tmp_path) -> None:
    present = chunk_file_path(tmp_path, "d1:0")
    present.parent.mkdir(parents=True)
    present.write_bytes(b"x")

    assert missing_chunk_files(tmp_path, ["d1:0", "d1:1"]) == ["d1:1"]


def _save_test_chunk(cache_dir: Path, meta_base: dict, chunk: RagChunk) -> int:
    torch = pytest.importorskip("torch", reason="CPU tensor serialization requires the optional torch dependency")

    encoded = KVChunk(
        chunk_id=chunk.chunk_id,
        token_ids=list(chunk.token_ids),
        kv_by_layer={"layer": torch.zeros(2, len(chunk.token_ids), 2, 4)},
    )
    save_chunk(
        chunk_file_path(cache_dir, chunk.chunk_id),
        meta=chunk_meta(meta_base, chunk),
        chunk=encoded,
    )
    return sum(int(tensor.nbytes) for tensor in encoded.kv_by_layer.values())


@pytest.fixture
def gpu_map_constructor(monkeypatch):
    """Mock the CUDA map boundary for CPU payload storage protocol tests."""
    chunk_map = Mock(spec=GPUChunkMap)
    chunk_map.device = "cuda:2"
    chunk_map.nbytes = 48
    constructor = Mock(return_value=chunk_map)
    monkeypatch.setattr("benchmarks.rag.kv_cache.GPUChunkMap", constructor)
    return constructor


def test_cpu_chunk_store_loads_full_corpus_upfront(tmp_path, gpu_map_constructor) -> None:
    meta_base = _meta_base()
    chunk_a = _chunk("d1", 0, [1, 2, 3, 4])
    chunk_b = _chunk("d1", 1, [5, 6, 7, 8])
    nbytes_a = _save_test_chunk(tmp_path, meta_base, chunk_a)
    nbytes_b = _save_test_chunk(tmp_path, meta_base, chunk_b)

    store = CpuChunkStore(
        tmp_path, meta_base=meta_base, chunks=[chunk_a, chunk_b], device="cuda:2"
    )
    gpu_map_constructor.assert_called_once_with(["d1:0", "d1:1"], device="cuda:2")
    chunk_map = gpu_map_constructor.return_value
    chunk_map.lookup.return_value.detach.return_value.cpu.return_value.tolist.return_value = [1, 0]

    fetched = store.fetch([chunk_b.chunk_id, chunk_a.chunk_id])
    assert [chunk.token_ids for chunk in fetched] == [[5, 6, 7, 8], [1, 2, 3, 4]]
    assert all(tensor.device.type == "cpu" for chunk in fetched for tensor in chunk.kv_by_layer.values())
    stats = store.stats()
    assert stats["kv_store_backend"] == "cpu"
    assert stats["kv_store_chunk_count"] == 2
    assert stats["kv_store_resident_bytes"] == nbytes_a + nbytes_b
    assert stats["kv_store_load_time_ms"] >= 0.0
    assert stats["kv_chunk_map_device"] == "cuda:2"
    assert stats["kv_chunk_map_bytes"] == 48

    chunk_map.lookup.side_effect = ValueError("Retrieved chunk ID has no pre-encoded KV chunk.")
    with pytest.raises(RuntimeError, match="not part of the corpus"):
        store.fetch(["unknown:0"])


def test_cpu_chunk_store_missing_files_raise_before_loading(tmp_path) -> None:
    meta_base = _meta_base()
    chunk_a = _chunk("d1", 0, [1, 2, 3, 4])
    chunk_b = _chunk("d1", 1, [5, 6, 7, 8])
    _save_test_chunk(tmp_path, meta_base, chunk_a)

    with pytest.raises(RuntimeError, match="precompute-kv stage"):
        CpuChunkStore(tmp_path, meta_base=meta_base, chunks=[chunk_a, chunk_b], device="cuda:0")


def test_cpu_chunk_store_stale_cache_raises_actionably(tmp_path, gpu_map_constructor) -> None:
    meta_base = _meta_base()
    chunk_a = _chunk("d1", 0, [1, 2, 3, 4])
    _save_test_chunk(tmp_path, meta_base, chunk_a)
    drifted = _chunk("d1", 0, [9, 9, 9, 9])

    with pytest.raises(RuntimeError, match="precompute-kv stage"):
        CpuChunkStore(tmp_path, meta_base=meta_base, chunks=[drifted], device="cuda:2")
    gpu_map_constructor.return_value.close.assert_called_once_with()


def _stub_payloads(tmp_path, monkeypatch, chunks: list[RagChunk]) -> Mock:
    encoded = [
        KVChunk(chunk_id=chunk.chunk_id, token_ids=chunk.token_ids, kv_by_layer={})
        for chunk in chunks
    ]
    for chunk in chunks:
        path = chunk_file_path(tmp_path, chunk.chunk_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"cache-presence-only")
    loader = Mock(side_effect=encoded)
    monkeypatch.setattr("benchmarks.rag.kv_cache.load_chunk", loader)
    return loader


def test_cpu_chunk_store_uses_one_gpu_lookup_for_ordered_duplicate_rows(
    tmp_path, monkeypatch, gpu_map_constructor
) -> None:
    chunks = [_chunk("d1", 0, [1, 2]), _chunk("d1", 1, [3, 4])]
    loader = _stub_payloads(tmp_path, monkeypatch, chunks)
    store = CpuChunkStore(tmp_path, meta_base=_meta_base(), chunks=chunks, device="cuda:2")
    chunk_map = gpu_map_constructor.return_value
    rows = chunk_map.lookup.return_value
    rows.detach.return_value.cpu.return_value.tolist.return_value = [1, 0, 1]

    fetched = store.fetch(["d1:1", "d1:0", "d1:1"])

    assert [chunk.chunk_id for chunk in fetched] == ["d1:1", "d1:0", "d1:1"]
    assert fetched[0] is fetched[2]
    chunk_map.lookup.assert_called_once_with(["d1:1", "d1:0", "d1:1"])
    rows.detach.return_value.cpu.assert_called_once_with()
    assert all(call.kwargs["device"] == "cpu" for call in loader.call_args_list)
    rows.detach.return_value.cpu.return_value.tolist.return_value = []
    assert store.fetch([]) == []

    store.close()
    chunk_map.close.assert_called_once_with()
    assert store.resident_bytes == 0
    assert store._entries == []


def test_cpu_chunk_store_does_not_fallback_when_cuda_map_fails(
    tmp_path, monkeypatch, gpu_map_constructor
) -> None:
    chunks = [_chunk()]
    loader = _stub_payloads(tmp_path, monkeypatch, chunks)
    gpu_map_constructor.side_effect = RuntimeError("CUDA allocation failed")

    with pytest.raises(RuntimeError, match="CUDA allocation failed"):
        CpuChunkStore(tmp_path, meta_base=_meta_base(), chunks=chunks, device="cuda:2")

    loader.assert_not_called()


def test_cpu_chunk_store_releases_map_after_partial_load_failure(
    tmp_path, monkeypatch, gpu_map_constructor
) -> None:
    chunks = [_chunk("d1", 0), _chunk("d1", 1)]
    loader = _stub_payloads(tmp_path, monkeypatch, chunks)
    loader.side_effect = [
        KVChunk(chunk_id="d1:0", token_ids=[1], kv_by_layer={}),
        None,
    ]

    with pytest.raises(RuntimeError, match="precompute-kv stage"):
        CpuChunkStore(tmp_path, meta_base=_meta_base(), chunks=chunks, device="cuda:2")

    gpu_map_constructor.return_value.close.assert_called_once_with()


def test_cpu_chunk_store_rejects_cpu_mapping_device(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires a CUDA device"):
        CpuChunkStore(tmp_path, meta_base=_meta_base(), chunks=[], device="cpu")


def test_cpu_chunk_store_cuda_lookup_with_cpu_payloads(tmp_path) -> None:
    torch = pytest.importorskip("torch", reason="CUDA lookup integration requires PyTorch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA lookup integration requires a CUDA device")
    chunks = [_chunk("d1", 0, [1, 2]), _chunk("d1", 1, [3, 4])]
    for chunk in chunks:
        _save_test_chunk(tmp_path, _meta_base(), chunk)

    store = CpuChunkStore(tmp_path, meta_base=_meta_base(), chunks=chunks, device="cuda:0")
    try:
        assert store._chunk_map.device.type == "cuda"
        fetched = store.fetch(["d1:1", "d1:0", "d1:1"])
        assert [chunk.chunk_id for chunk in fetched] == ["d1:1", "d1:0", "d1:1"]
        assert all(tensor.device.type == "cpu" for chunk in fetched for tensor in chunk.kv_by_layer.values())
        with pytest.raises(RuntimeError, match="not part of the corpus"):
            store.fetch(["unknown:0"])
    finally:
        store.close()
