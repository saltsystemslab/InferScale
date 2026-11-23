from __future__ import annotations

from benchmarks.common.embedding_cache import CachedEmbedder
from benchmarks.rag.preembed import preembed_rag_embeddings


def test_preembed_caches_under_the_purposes_of_memory_runs_so_both_share_one_cache(
    multihop_config, embedding_api, monkeypatch
):
    purposes: list[str] = []
    embed_batch = CachedEmbedder.embed_batch

    def embed_batch_recording_its_purpose(self, texts, memory_action):
        purposes.append(memory_action)
        return embed_batch(self, texts, memory_action)

    monkeypatch.setattr(CachedEmbedder, "embed_batch", embed_batch_recording_its_purpose)

    preembed_rag_embeddings(multihop_config(embed_batch_size=100))

    assert purposes == ["add", "search"]
    assert purposes == [CachedEmbedder.DOCUMENT_PURPOSE, CachedEmbedder.QUERY_PURPOSE]
