from __future__ import annotations

import numpy as np
import pytest

from inferscale.v1.kv.chunk_store import build_chunk_store, chunk_nbytes, register_chunks
from inferscale.v1.kv.compose import memory_parts, reverse_ranked_ids
from inferscale.v1.kv.corpus import KVCorpus
from inferscale.v1.kv.memory_store import GPUMemoryStore
from inferscale.v1.kv.packed_memory_store import PackedGPUMemoryStore
from inferscale.v1.kv.serialization import (
    CONTEXT_IDS_KEY,
    chunk_from_payload,
    chunk_to_payload,
    load_torch_payload,
    save_torch_payload,
)
from inferscale.v1.types import KVChunk, ScaffoldChunks, SearchHit


def _chunk(chunk_id: str, tokens: list[int]) -> KVChunk:
    kv = np.asarray(tokens, dtype=np.float32).reshape(1, -1, 1, 1)
    return KVChunk(chunk_id=chunk_id, token_ids=list(tokens), kv_by_layer={"l0": np.repeat(kv, 2, axis=0)})


def test_reverse_ranked_ids_dedups_and_puts_the_best_hit_last() -> None:
    hits = [
        SearchHit("a", {}, 0.9, -0.9, 1),
        SearchHit("b", {}, 0.8, -0.8, 2),
        SearchHit("a", {}, 0.7, -0.7, 3),
        SearchHit("c", {}, 0.6, -0.6, 4),
    ]
    assert reverse_ranked_ids(hits) == ["c", "b", "a"]


def test_memory_parts_uses_the_empty_chunk_when_nothing_is_selected() -> None:
    scaffold = ScaffoldChunks(header=_chunk("h", [1]), empty=_chunk("e", [2]), footer=_chunk("f", [3]))
    assert [part.chunk_id for part in memory_parts(scaffold, [])] == ["h", "e", "f"]
    assert [part.chunk_id for part in memory_parts(scaffold, [_chunk("x", [4])])] == ["h", "x", "f"]


def test_build_chunk_store_backends() -> None:
    assert isinstance(build_chunk_store("gpu", device="cpu", top_k=3), GPUMemoryStore)
    assert isinstance(
        build_chunk_store("gpu", device="cpu", top_k=3, device_selection=True), PackedGPUMemoryStore
    )
    with pytest.raises(ValueError):
        build_chunk_store("disk", device="cpu", top_k=3)


def test_register_chunks_requires_matching_ids() -> None:
    store = GPUMemoryStore(device="cpu")
    with pytest.raises(ValueError, match="does not match chunk_id"):
        register_chunks(store, {"other": _chunk("c", [1])})


def test_corpus_moves_tensors_into_the_store_and_fetches_by_id() -> None:
    corpus = KVCorpus(GPUMemoryStore(device="cpu"))
    chunk = _chunk("c1", [5, 6, 7])
    expected_bytes = chunk_nbytes(chunk)
    corpus.add(chunk)
    corpus.add(_chunk("c2", [8]))

    assert chunk.kv_by_layer == {}
    assert corpus.ids() == ["c1", "c2"]
    assert "c1" in corpus and len(corpus) == 2
    assert corpus.get("c1").kv_by_layer == {}
    assert corpus.get("c1").token_ids == [5, 6, 7]

    fetched = corpus.fetch(["c2", "c1"])
    assert [item.chunk_id for item in fetched] == ["c2", "c1"]
    assert fetched[1].kv_by_layer["l0"].shape == (2, 3, 1, 1)
    corpus.release(["c2", "c1"])

    stats = corpus.stats()
    assert stats["chunk_count"] == 2
    assert stats["chunk_bytes"] == expected_bytes + chunk_nbytes(_chunk("c2", [8]))
    assert stats["store_num_users"] == 2

    with pytest.raises(ValueError, match="already in the corpus"):
        corpus.add(_chunk("c1", [1]))
    with pytest.raises(RuntimeError, match="no pre-encoded KV chunk"):
        corpus.fetch(["missing"])

    corpus.close()
    assert corpus.ids() == []
    assert corpus.store.get_all_user_ids() == []


def test_chunk_payload_round_trip_keeps_the_frozen_keys(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    chunk = KVChunk(
        chunk_id="c",
        token_ids=[1, 2],
        kv_by_layer={"model.layers.0.self_attn.attn": torch.ones(2, 2, 1, 4)},
        context_ids=("t1",),
        context_prefix_tokens=3,
        raw_context_prefix_tokens=5,
        context_prefix_truncated_tokens=2,
    )
    payload = chunk_to_payload(chunk)
    assert set(payload) == {
        "token_ids",
        "context_turn_ids",
        "context_prefix_tokens",
        "raw_context_prefix_tokens",
        "context_prefix_truncated_tokens",
        "kv_by_layer",
    }
    assert CONTEXT_IDS_KEY == "context_turn_ids"
    assert payload["context_turn_ids"] == ["t1"]

    path = tmp_path / "chunk.pt"
    save_torch_payload(path, {"version": 1, "chunk": payload})
    loaded = load_torch_payload(path)
    restored = chunk_from_payload("c", loaded["chunk"], device="cpu")
    assert restored.token_ids == [1, 2]
    assert restored.context_ids == ("t1",)
    assert restored.context_prefix_tokens == 3
    assert restored.raw_context_prefix_tokens == 5
    assert restored.context_prefix_truncated_tokens == 2
    assert torch.equal(restored.kv_by_layer["model.layers.0.self_attn.attn"], torch.ones(2, 2, 1, 4))
    assert load_torch_payload(tmp_path / "missing.pt") is None


def test_compose_concatenates_values_and_rotates_keys_at_new_positions() -> None:
    torch = pytest.importorskip("torch")
    from inferscale.v1.kv.encoder import ChunkedRopeEncoder
    from inferscale.v1.kv.rope import rotate_pre_rope_k

    head_dim, heads, max_position = 4, 2, 8
    positions = torch.arange(max_position, dtype=torch.float32)
    freqs = torch.outer(positions, torch.tensor([1.0, 0.5]))
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_table, sin_table = emb.cos(), emb.sin()
    encoder = ChunkedRopeEncoder.from_tables(
        model="m", device="cpu", max_position=max_position, tokenizer=None,
        cos_table=cos_table, sin_table=sin_table,
    )
    layer = "model.layers.0.self_attn.attn"
    first = KVChunk("a", [1, 2, 3], {layer: torch.randn(2, 3, heads, head_dim)})
    second = KVChunk("b", [4, 5], {layer: torch.randn(2, 2, heads, head_dim)})

    composed = encoder.compose([first, second])[layer]
    assert composed.shape == (2, 5, heads, head_dim)
    values = torch.cat([first.kv_by_layer[layer][1], second.kv_by_layer[layer][1]], dim=0)
    assert torch.equal(composed[1], values)
    k_pre = torch.cat([first.kv_by_layer[layer][0], second.kv_by_layer[layer][0]], dim=0)
    expected = rotate_pre_rope_k(k_pre.transpose(0, 1).contiguous(), cos_table[:5], sin_table[:5]).transpose(0, 1)
    assert torch.allclose(composed[0], expected)

    with pytest.raises(RuntimeError, match="exceeding max_position"):
        encoder.compose([KVChunk("x", list(range(9)), {layer: torch.randn(2, 9, heads, head_dim)})])
    with pytest.raises(RuntimeError, match="released"):
        encoder.encode("x", [1])
