from __future__ import annotations

from benchmarks.rag import kv_precompute
from benchmarks.rag.corpus import load_rag_corpus
from benchmarks.rag.kv_cache import RagChunkCache
from benchmarks.rag.kv_precompute import precompute_rag_kv


def cache_of(config) -> RagChunkCache:
    return RagChunkCache.for_config(config, load_rag_corpus(config).docs)


def test_precompute_encodes_each_chunk_after_the_chunks_preceding_it_in_its_document(
    multihop_config, rope_encoders
):
    config = multihop_config(context_window=2)
    corpus = load_rag_corpus(config)

    precompute_rag_kv(config)

    [encoder] = rope_encoders
    assert encoder.settings == {
        "model": "llama",
        "dtype": "bfloat16",
        "device": "cpu",
        "max_position": 32768,
    }
    assert [plan.chunk_id for plan in encoder.encoded_plans] == [
        chunk.chunk_id for chunk in corpus.chunks
    ]
    first, second, third, fourth = encoder.encoded_plans[:4]
    doc_id = corpus.chunks[0].doc_id
    assert first.context_ids == ()
    assert second.context_ids == (f"{doc_id}:0",)
    assert third.context_ids == (f"{doc_id}:0", f"{doc_id}:1")
    assert fourth.context_ids == (f"{doc_id}:1", f"{doc_id}:2")
    assert fourth.input_token_ids == (
        corpus.chunks[1].token_ids + corpus.chunks[2].token_ids + corpus.chunks[3].token_ids
    )
    assert encoder.closed


def test_only_the_missing_chunks_are_encoded_again(multihop_config, rope_encoders):
    config = multihop_config()
    corpus = load_rag_corpus(config)
    precompute_rag_kv(config)
    rope_encoders.clear()
    cache = cache_of(config)
    lost = [corpus.chunks[4], corpus.chunks[20]]
    for chunk in lost:
        cache.chunk_path(chunk.chunk_id).unlink()

    summary = precompute_rag_kv(config)

    [encoder] = rope_encoders
    assert [plan.chunk_id for plan in encoder.encoded_plans] == [chunk.chunk_id for chunk in lost]
    assert encoder.encoded_scaffold == {}
    assert (summary["encoded"], summary["skipped"]) == (2, 22)
    assert encoder.closed


def test_allocator_cache_is_released_every_25_encodes_to_keep_fragmentation_from_growing(
    multihop_config, rope_encoders, monkeypatch, rag_log
):
    encodes_at_release: list[int] = []
    monkeypatch.setattr(
        kv_precompute,
        "empty_cuda_cache",
        lambda: encodes_at_release.append(len(rope_encoders[0].encoded_plans)),
    )

    precompute_rag_kv(multihop_config(chunk_size=8))

    assert encodes_at_release == [25, 50, 75, 100, 125, 141]
    progress = [message for _, message in rag_log if message.startswith("Encoded")]
    assert progress == [
        f"Encoded {encoded}/141 missing RAG KV chunks" for encoded in encodes_at_release
    ]
