from __future__ import annotations

import pytest

from inferscale.v1.kv.corpus import KVCorpus
from inferscale.v1.kv.memory_store import GPUMemoryStore
from inferscale.v1.types import KVChunk


class TracedTensor:
    def __init__(self, nbytes: int, device: str = "cuda:0", history: tuple[str, ...] = ()) -> None:
        self.nbytes = nbytes
        self.device = device
        self.history = history

    def contiguous(self) -> TracedTensor:
        return TracedTensor(self.nbytes, self.device, (*self.history, "contiguous"))


def test_gpu_resident_store_has_nothing_to_stage_or_meter() -> None:
    store = GPUMemoryStore(device="cuda:0")
    store.add_user_memory("alice", {"layer": TracedTensor(8)}, 2)

    assert GPUMemoryStore.num_staging_slots == 0
    assert store.prefetch_user_to_gpu("alice") is False
    assert store.release_staging("alice") is None
    assert store.finalize_packed() is None
    assert store.get_bench_summary() == {}
    assert store.transfer_totals() == {}
    assert store.get_bench_summary() is not store.get_bench_summary()
    assert store.transfer_totals() is not store.transfer_totals()


class RefusingStore:
    def add_user_memory(
        self, user_id: str, kv_by_layer: dict, num_tokens: int, token_ids: list
    ) -> None:
        raise RuntimeError("Cannot add chunks after the packed GPU store is finalized.")

    def get_stats(self) -> dict:
        return {}


def test_corpus_does_not_count_a_chunk_the_store_refused() -> None:
    corpus = KVCorpus(RefusingStore())

    with pytest.raises(RuntimeError, match="finalized"):
        corpus.add(KVChunk("refused", [1], {"layer": TracedTensor(5000)}))

    assert corpus.stats() == {"chunk_count": 0, "chunk_bytes": 0, "layers": 0}
