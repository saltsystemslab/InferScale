from __future__ import annotations

from typing import Any

import pytest
from library_fakes import FakeChunkStore, FakeEncoder

from benchmarks.memory import composer as composer_module
from benchmarks.memory.composer import SampleComposer
from benchmarks.memory.composer_stats import precompute_stats
from benchmarks.memory.data import ConversationSample, Turn
from benchmarks.memory.mem0.fact_catalog import MemoryFact, fact_catalog_hits
from inferscale.v1.kv.chunk_store import chunk_nbytes
from inferscale.v1.kv.memory_store import GPUMemoryStore
from inferscale.v1.types import KVChunk, SearchHit

PARSE_HITS_MS = 1
PLAN_MEMORY_MS = 3
FETCH_CHUNKS_MS = 7
COMPOSE_MS = 20


class Clock:
    def __init__(self) -> None:
        self.now_s = 100.0

    def perf_counter(self) -> float:
        return self.now_s

    def advance_ms(self, milliseconds: float) -> None:
        self.now_s += milliseconds / 1000


class ChunkStore(FakeChunkStore):
    def __init__(self, clock: Clock | None = None, *, staging_slots: int = 0) -> None:
        super().__init__()
        self.clock = clock
        self.num_staging_slots = staging_slots
        self.lookup_finalizations = 0
        self.failing_finalizations = 0
        self.released: list[str] = []

    def finalize_chunk_lookup(self) -> None:
        self.lookup_finalizations += 1
        if self.failing_finalizations > 0:
            self.failing_finalizations -= 1
            raise RuntimeError("GPU lookup allocation failed")
        super().finalize_chunk_lookup()

    def get_chunk_memories(self, chunk_ids: Any) -> Any:
        if self.clock is not None:
            self.clock.advance_ms(FETCH_CHUNKS_MS)
        return super().get_chunk_memories(chunk_ids)

    def release_staging(self, user_id: str) -> None:
        self.released.append(user_id)

    def stored_layers(self) -> dict[str, list[str]]:
        return {
            chunk_id: sorted(GPUMemoryStore.get_user_memory(self, chunk_id).kv_by_layer)
            for chunk_id in self.get_all_user_ids()
        }


def _encoder(max_position: int = 4096) -> FakeEncoder:
    return FakeEncoder(
        model="fake-model", dtype="bfloat16", device="cpu", max_position=max_position
    )


def _sample() -> ConversationSample:
    turns = [
        Turn(
            sample_id="sample-1",
            session_id="session_1",
            session_index=1,
            turn_index=turn_index,
            speaker="Alice",
            text=f"I like drink {turn_index}.",
            timestamp="2026-01-02",
        )
        for turn_index in range(3)
    ]
    return ConversationSample(sample_id="sample-1", turns=turns, qa=[], raw={})


def _fact(index: int) -> MemoryFact:
    return MemoryFact(
        id=f"fact-{index}",
        text=f"Alice likes drink {index}.",
        created_at="2026-01-02T00:00:00+00:00",
        timestamp_epoch=1767312000,
        sample_id="sample-1",
        source_session_index=1,
        source_session_id="session_1",
        source_turn_index=index,
        source_turn_id=f"sample-1:session_1:{index}",
        speaker="Alice",
        role="user",
    )


def _catalog_hits() -> list[SearchHit]:
    return fact_catalog_hits([_fact(0), _fact(1), _fact(2)])


def _encoded_composer(
    store: ChunkStore | None = None, *, hits: list[SearchHit] | None = None
) -> SampleComposer:
    composer = SampleComposer(encoder=_encoder(), chunk_store=store)
    composer.encode_sample(_sample(), _catalog_hits() if hits is None else hits)
    return composer


def test_chunks_moved_to_the_store_leave_metadata_behind_and_keep_their_measured_size() -> None:
    store = ChunkStore()
    composer = _encoded_composer(store)
    token_ids = {chunk_id: list(chunk.token_ids) for chunk_id, chunk in composer.chunks.items()}
    tensor_bytes = sum(chunk_nbytes(chunk) for chunk in composer.chunks.values())
    assert tensor_bytes > 0
    assert (composer.chunks_in_store, composer.stored_chunk_bytes) == (False, 0)

    composer.move_chunks_to_store()
    composer.move_chunks_to_store()

    assert composer.chunks_in_store is True
    assert composer.stored_chunk_bytes == tensor_bytes
    assert composer.stored_chunk_layers == len(FakeEncoder.layers)
    assert {chunk_id: chunk.token_ids for chunk_id, chunk in composer.chunks.items()} == token_ids
    assert all(chunk.kv_by_layer == {} for chunk in composer.chunks.values())
    assert store.stored_layers() == dict.fromkeys(token_ids, sorted(FakeEncoder.layers))
    assert store.lookup_finalizations == 2


def test_scaffold_chunks_stay_with_the_composer_when_the_fact_chunks_move_to_the_store() -> None:
    store = ChunkStore()
    composer = _encoded_composer(store)
    scaffold_bytes = [chunk_nbytes(chunk) for chunk in composer.encoded_scaffold_chunks()]

    composer.move_chunks_to_store()
    composer.compose(_catalog_hits())

    assert [chunk_nbytes(chunk) for chunk in composer.encoded_scaffold_chunks()] == scaffold_bytes
    assert all(tensor_bytes > 0 for tensor_bytes in scaffold_bytes)
    assert store.get_all_user_ids() == ["fact-0", "fact-1", "fact-2"]
    assert store.lookup_batches == [["fact-2", "fact-1", "fact-0"]]


def test_failed_lookup_finalization_keeps_the_chunk_payloads_for_a_retry() -> None:
    store = ChunkStore()
    store.failing_finalizations = 1
    composer = _encoded_composer(store)
    tensor_bytes = sum(chunk_nbytes(chunk) for chunk in composer.chunks.values())

    with pytest.raises(RuntimeError, match="GPU lookup allocation failed"):
        composer.move_chunks_to_store()
    composer.move_chunks_to_store()

    assert composer.stored_chunk_bytes == tensor_bytes
    assert composer.stored_chunk_layers == len(FakeEncoder.layers)
    assert store.stored_layers() == dict.fromkeys(composer.chunks, sorted(FakeEncoder.layers))
    composed = composer.compose(_catalog_hits()[:1])
    assert composed.kv_by_layer[FakeEncoder.layers[0]].shape[1] == composed.num_tokens


def test_staged_facts_are_released_even_when_composing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ChunkStore()
    composer = _encoded_composer(store)

    def fail(chunks: list[KVChunk]) -> None:
        raise RuntimeError("compose kernel failed")

    monkeypatch.setattr(composer.encoder, "compose", fail)

    with pytest.raises(RuntimeError, match="compose kernel failed"):
        composer.compose(_catalog_hits()[:2])

    assert store.released == ["fact-1", "fact-0"]


@pytest.mark.parametrize("hits, top_k", [(_catalog_hits(), 3)])
def test_composer_built_without_a_store_composes_through_a_gpu_store(
    monkeypatch: pytest.MonkeyPatch, hits: list[SearchHit], top_k: int
) -> None:
    store = ChunkStore()
    requests: list[tuple[str, dict[str, Any]]] = []

    def build_chunk_store(backend: str, **options: Any) -> ChunkStore:
        requests.append((backend, options))
        return store

    monkeypatch.setattr(composer_module, "build_chunk_store", build_chunk_store)
    composer = _encoded_composer(hits=hits)

    composer.compose(hits)
    composer.compose(hits)

    assert requests == [("gpu", {"device": "cpu", "top_k": top_k})]
    assert composer.chunk_store is store
    assert composer.chunks_in_store is True
    assert store.lookup_finalizations == 2


def test_compose_time_runs_from_parsing_the_hits_to_the_composed_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr(composer_module, "time", clock)
    composer = _encoded_composer(ChunkStore(clock))

    def taking(milliseconds: float, step: Any) -> Any:
        def timed_step(*args: Any, **options: Any) -> Any:
            clock.advance_ms(milliseconds)
            return step(*args, **options)

        return timed_step

    monkeypatch.setattr(
        composer_module,
        "unique_memory_facts",
        taking(PARSE_HITS_MS, composer_module.unique_memory_facts),
    )
    monkeypatch.setattr(
        composer_module, "plan_kv_memory", taking(PLAN_MEMORY_MS, composer_module.plan_kv_memory)
    )
    monkeypatch.setattr(composer.encoder, "compose", taking(COMPOSE_MS, composer.encoder.compose))

    composed = composer.compose(_catalog_hits())

    assert composed.compose_time_ms == pytest.approx(
        PARSE_HITS_MS + PLAN_MEMORY_MS + FETCH_CHUNKS_MS + COMPOSE_MS
    )


def test_statistics_of_local_chunks_measure_their_tensors() -> None:
    composer = _encoded_composer()
    fact_bytes = sum(chunk_nbytes(chunk) for chunk in composer.chunks.values())
    scaffold_bytes = sum(chunk_nbytes(chunk) for chunk in composer.encoded_scaffold_chunks())
    tokens = sum(
        len(chunk.token_ids)
        for chunk in [*composer.chunks.values(), *composer.encoded_scaffold_chunks()]
    )

    stats = precompute_stats(composer)

    assert stats["kv_chunk_cache_residency"] == "gpu"
    assert stats["kv_precomputed_chunks"] == stats["llama_kv_chunk_count"] == 3
    assert stats["kv_precomputed_chunks_with_prefix"] == 6
    assert stats["kv_precomputed_tokens"] == tokens
    assert stats["kv_precomputed_layers"] == 2
    assert stats["kv_precomputed_devices"] == "cpu"
    assert stats["llama_kv_chunk_tensor_gpu_bytes"] == fact_bytes
    assert stats["llama_kv_prefix_tensor_gpu_bytes"] == scaffold_bytes
    assert stats["llama_kv_total_tensor_gpu_bytes"] == fact_bytes + scaffold_bytes
    assert stats["kv_precomputed_gpu_mb"] == (fact_bytes + scaffold_bytes) / (1024 * 1024)
    assert stats["llama_kv_total_tensor_gpu_mb"] == stats["kv_precomputed_gpu_mb"]
    assert stats["llama_kv_chunk_map_gpu_bytes"] == 0
    assert stats["llama_kv_chunk_metadata_cpu_bytes"] > 0
    assert not any(name.startswith("kv_chunk_store_") for name in stats)


@pytest.mark.parametrize(
    "staging_slots, residency, devices",
    [(0, "gpu", "cpu"), (4, "cpu", "cpu,cpu(pinned)")],
)
def test_statistics_of_stored_chunks_report_the_sizes_measured_before_the_move(
    staging_slots: int, residency: str, devices: str
) -> None:
    store = ChunkStore(staging_slots=staging_slots)
    composer = _encoded_composer(store)
    local = precompute_stats(composer)

    composer.move_chunks_to_store()
    stored = precompute_stats(composer)

    assert stored["kv_chunk_cache_residency"] == residency
    assert stored["kv_precomputed_devices"] == devices
    assert local["kv_chunk_cache_residency"] == "gpu"
    for name in (
        "kv_precomputed_chunks",
        "kv_precomputed_chunks_with_prefix",
        "kv_precomputed_tokens",
        "kv_precomputed_layers",
        "kv_precomputed_gpu_mb",
        "llama_kv_chunk_tensor_gpu_bytes",
        "llama_kv_prefix_tensor_gpu_bytes",
        "llama_kv_total_tensor_gpu_bytes",
    ):
        assert stored[name] == local[name]
    assert stored["kv_chunk_store_num_users"] == 3
    assert stored["kv_chunk_store_total_tokens"] == sum(
        len(chunk.token_ids) for chunk in composer.chunks.values()
    )
