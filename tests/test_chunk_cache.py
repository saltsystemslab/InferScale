from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from locomo_jasper_bench.data import ConversationSample, Turn
from locomo_jasper_bench.kv.chunk_cache import (
    CHUNK_CACHE_VERSION,
    cache_meta,
    cache_path_for,
    chunk_cache_dir,
    facts_fingerprint,
    payload_is_valid,
    scaffold_chunks_match,
)
from locomo_jasper_bench.kv.types import EncodedChunk, MemoryFact


def _fact(memory_id: str, text_hash: str = "hash", turn_id: str = "turn-1") -> MemoryFact:
    return MemoryFact(
        memory_id=memory_id,
        text=f"text for {memory_id}",
        text_hash=text_hash,
        created_at="2023-05-08",
        source_session_index=1,
        source_session_id="session-1",
        source_turn_index=2,
        source_turn_id=turn_id,
    )


def _sample(sample_id: str = "conv-1") -> ConversationSample:
    return ConversationSample(
        sample_id=sample_id,
        turns=[
            Turn(
                sample_id=sample_id,
                session_id="session-1",
                session_index=1,
                turn_index=1,
                speaker="Caroline",
                text="hello",
            )
        ],
        qa=[],
        raw={},
    )


def _scaffold(
    header: list[int],
    footer: list[int],
    memory_list_header: list[int] | None = None,
    empty_memory: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        header_token_ids=header,
        footer_token_ids=footer,
        memory_list_header_token_ids=memory_list_header,
        empty_memory_token_ids=empty_memory or [9],
    )


def _chunk(token_ids: list[int]) -> EncodedChunk:
    return EncodedChunk(token_ids=token_ids, kv_by_layer={"layer": SimpleNamespace(nbytes=8)})


def test_facts_fingerprint_is_deterministic_and_content_sensitive() -> None:
    facts = [_fact("a"), _fact("b")]

    assert facts_fingerprint(facts) == facts_fingerprint([_fact("a"), _fact("b")])
    assert facts_fingerprint(facts) != facts_fingerprint([_fact("b"), _fact("a")])
    assert facts_fingerprint(facts) != facts_fingerprint([_fact("a"), _fact("b", text_hash="other")])
    assert facts_fingerprint(facts) != facts_fingerprint([_fact("a"), _fact("b", turn_id="turn-9")])
    assert len(facts_fingerprint(facts)) == 20


def test_cache_path_encodes_every_key_component(tmp_path: Path) -> None:
    path = cache_path_for(
        model="Qwen/Qwen3-14B",
        dtype="bfloat16",
        context_window=50,
        max_position=32768,
        block_size=16,
        sample=_sample(),
        facts=[_fact("a")],
        cache_root=tmp_path,
    )

    assert path.suffix == ".pt"
    assert f"v{CHUNK_CACHE_VERSION}" in path.parts
    assert "Qwen__Qwen3-14B" in path.parts
    assert "bfloat16-w50-p32768-b16" in path.parts
    assert path.name.startswith("conv-1-")
    # Different facts must produce a different file in the same directory.
    other = cache_path_for(
        model="Qwen/Qwen3-14B",
        dtype="bfloat16",
        context_window=50,
        max_position=32768,
        block_size=16,
        sample=_sample(),
        facts=[_fact("a", text_hash="changed")],
        cache_root=tmp_path,
    )
    assert other.parent == path.parent
    assert other.name != path.name
    assert chunk_cache_dir(
        model="Qwen/Qwen3-14B",
        dtype="bfloat16",
        context_window=50,
        max_position=32768,
        block_size=16,
        cache_root=tmp_path,
    ) == path.parent


def _valid_payload(meta: dict) -> dict:
    return {
        "version": CHUNK_CACHE_VERSION,
        "meta": dict(meta),
        "cos_table": SimpleNamespace(),
        "sin_table": SimpleNamespace(),
        "fact_chunks": {"a": {"token_ids": [1]}},
        "scaffold_chunks": {
            "header": {"token_ids": [1]},
            "memory_list_header": None,
            "empty_memory": {"token_ids": [9]},
            "footer": {"token_ids": [2]},
        },
    }


def test_payload_validation_rejects_mismatches() -> None:
    meta = cache_meta(
        model="m",
        dtype="bfloat16",
        context_window=0,
        max_position=32768,
        block_size=16,
        sample=_sample(),
        facts=[_fact("a")],
    )
    payload = _valid_payload(meta)

    assert payload_is_valid(payload, expected_meta=meta, expected_fact_ids=["a"])
    assert not payload_is_valid(payload, expected_meta=meta, expected_fact_ids=["a", "b"])
    assert not payload_is_valid(
        payload, expected_meta={**meta, "context_window": 5}, expected_fact_ids=["a"]
    )
    assert not payload_is_valid(
        {**payload, "version": CHUNK_CACHE_VERSION + 1}, expected_meta=meta, expected_fact_ids=["a"]
    )
    headerless = _valid_payload(meta)
    headerless["scaffold_chunks"]["header"] = None
    assert not payload_is_valid(headerless, expected_meta=meta, expected_fact_ids=["a"])
    assert not payload_is_valid("not-a-dict", expected_meta=meta, expected_fact_ids=["a"])


def test_scaffold_chunks_match_compares_all_slots() -> None:
    chunks = {
        "header": _chunk([1, 2]),
        "memory_list_header": None,
        "empty_memory": _chunk([9]),
        "footer": _chunk([3]),
    }

    assert scaffold_chunks_match(chunks, _scaffold([1, 2], [3]))
    assert not scaffold_chunks_match(chunks, _scaffold([1, 2, 4], [3]))
    assert not scaffold_chunks_match(chunks, _scaffold([1, 2], [3], memory_list_header=[7]))
    assert not scaffold_chunks_match({**chunks, "footer": None}, _scaffold([1, 2], [3]))


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from locomo_jasper_bench.kv.chunk_cache import load_sample_chunks, save_sample_chunks

    sample = _sample()
    facts = [_fact("a"), _fact("b")]
    meta = cache_meta(
        model="m",
        dtype="bfloat16",
        context_window=0,
        max_position=64,
        block_size=16,
        sample=sample,
        facts=facts,
    )
    path = tmp_path / "conv-1.pt"

    def chunk(seed: int) -> EncodedChunk:
        return EncodedChunk(
            token_ids=[seed, seed + 1],
            kv_by_layer={"model.layers.0.self_attn.attn": torch.full((2, 2), float(seed))},
            context_turn_ids=("turn-1",),
            context_prefix_tokens=seed,
        )

    save_sample_chunks(
        path,
        meta=meta,
        fact_chunks={"a": chunk(1), "b": chunk(2)},
        scaffold_chunks={
            "header": chunk(3),
            "memory_list_header": None,
            "empty_memory": chunk(4),
            "footer": chunk(5),
        },
        cos_table=torch.ones(4, 2),
        sin_table=torch.zeros(4, 2),
    )

    loaded = load_sample_chunks(
        path, device="cpu", expected_meta=meta, expected_fact_ids=["a", "b"]
    )
    assert loaded is not None
    assert loaded.fact_chunks["a"].token_ids == [1, 2]
    assert loaded.fact_chunks["a"].context_turn_ids == ("turn-1",)
    assert torch.equal(
        loaded.fact_chunks["b"].kv_by_layer["model.layers.0.self_attn.attn"],
        torch.full((2, 2), 2.0),
    )
    assert loaded.scaffold_chunks["memory_list_header"] is None
    assert loaded.scaffold_chunks["footer"].token_ids == [5, 6]
    assert torch.equal(loaded.cos_table, torch.ones(4, 2))

    # Any key drift is a clean miss, not an exception.
    assert (
        load_sample_chunks(
            path,
            device="cpu",
            expected_meta={**meta, "facts_fingerprint": "different"},
            expected_fact_ids=["a", "b"],
        )
        is None
    )
    assert (
        load_sample_chunks(
            path, device="cpu", expected_meta=meta, expected_fact_ids=["a"]
        )
        is None
    )
    missing = load_sample_chunks(
        tmp_path / "absent.pt", device="cpu", expected_meta=meta, expected_fact_ids=["a"]
    )
    assert missing is None

    corrupted = tmp_path / "corrupted.pt"
    corrupted.write_bytes(b"not a torch file")
    assert (
        load_sample_chunks(
            corrupted, device="cpu", expected_meta=meta, expected_fact_ids=["a", "b"]
        )
        is None
    )