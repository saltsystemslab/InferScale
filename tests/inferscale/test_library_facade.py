from __future__ import annotations

import pytest

import inferscale.v1.api as api_module
from inferscale.v1 import Chunk, InferScale, InferScaleConfig
from inferscale.v1.kv.registry import namespace_stats
from library_fakes import FakeChunkStore, FakeEmbedder, FakeEncoder, FakeEngine, FakeIndex


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setattr(api_module, "ChunkedRopeEncoder", FakeEncoder)
    config = InferScaleConfig.from_dict({"model": "fake-model", "top_k": 2, "engine": {"block_size": 4}})
    facade = InferScale(
        config,
        embedder=FakeEmbedder(),
        index=FakeIndex(),
        chunk_store=FakeChunkStore(),
        engine=FakeEngine(),
    )
    yield facade
    facade.close()


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", text="Alice has a cat named Miso.\n"),
        Chunk(id="c2", text="Bob teaches chemistry.\n", token_ids=[70, 71, 72]),
        Chunk(id="c3", text="Alice lives in Berlin.\n", context_token_ids=[5, 6], context_ids=("c1",)),
    ]


def test_query_requires_start(engine) -> None:
    with pytest.raises(RuntimeError, match="start\\(\\)"):
        engine.query("Who has a cat?")
    with pytest.raises(RuntimeError, match="Precompute at least one chunk"):
        engine.start()


def test_precompute_links_kv_chunks_and_index_entries_by_id(engine) -> None:
    stats = engine.precompute(_chunks())
    assert stats.chunk_count == 3
    assert stats.store["chunk_count"] == 3
    assert engine.chunk_ids() == ["c1", "c2", "c3"]
    assert engine._index.ids == ["c1", "c2", "c3"]
    assert engine._index.payloads[0] == {"text": "Alice has a cat named Miso.\n"}

    info = engine.get_chunk("c2")
    assert info.token_ids == [70, 71, 72]
    assert engine._corpus.get("c2").token_ids == [70, 71, 72]
    assert engine.get_chunk("c3").context_prefix_tokens == 2
    assert engine.get_chunk("c3").context_ids == ("c1",)
    tokenized = engine.get_chunk("c1").token_ids
    assert tokenized == engine.tokenizer.encode("Alice has a cat named Miso.\n\n", add_special_tokens=False)
    assert engine._corpus.scaffold is not None

    with pytest.raises(ValueError, match="Duplicate chunk id"):
        engine.precompute([Chunk(id="c1", text="again")])
    assert engine.precompute([]).chunk_count == 0


def test_phase_order_and_query_flow(engine) -> None:
    engine.precompute(_chunks())
    engine.start()
    engine.start()  # idempotent
    fake_engine = engine._engine
    assert fake_engine.started_with["kv_connector"] == "MemoryKVConnector"
    assert fake_engine.started_with["kv_connector_module_path"] == "inferscale.v1.kv.connector"
    extra = fake_engine.started_with["kv_connector_extra_config"]
    assert extra["memory_namespace"] == engine._namespace
    assert extra["default_user_id"] == engine._active_user_id
    assert extra["memory_store_backend"] == "gpu"
    assert engine._encoder.released is True
    assert engine._index.finalized is True
    assert engine._corpus.store.lookup_finalized is True

    with pytest.raises(RuntimeError, match="serving"):
        engine.precompute([Chunk(id="late", text="late")])

    result = engine.query("Who has a cat named Miso?")
    assert len(result.hits) == 2
    assert result.text.startswith("answer over")
    assert result.ttft_ms == 12.0
    assert result.retrieval is not None
    selected = result.metrics["kv_selected_chunk_ids"]
    assert selected == list(reversed([hit.id for hit in result.hits]))
    assert engine._corpus.store.lookup_batches[-1] == selected
    assert engine._encoder.composed[-1] == ["scaffold:header", *selected, "scaffold:footer"]

    scaffold = engine._corpus.scaffold
    memory_tokens = (
        len(scaffold.header.token_ids)
        + sum(len(engine.get_chunk(chunk_id).token_ids) for chunk_id in selected)
        + len(scaffold.footer.token_ids)
    )
    assert result.metrics["kv_memory_tokens"] == memory_tokens
    assert fake_engine.prompts[-1][:memory_tokens] == scaffold.header.token_ids + [
        token for chunk_id in selected for token in engine.get_chunk(chunk_id).token_ids
    ] + scaffold.footer.token_ids
    assert result.metrics["kv_loaded_memory_tokens"] % 4 == 0
    assert result.metrics["kv_block_size"] == 4
    assert result.metrics["kv_engine_time_to_first_token_ms"] == 12.0
    assert result.metrics["answer_generate_time_ms"] == pytest.approx(250.0)
    assert "query_to_first_token_ms" in result.metrics
    assert result.metrics["query_embedding_time_ms"] == result.retrieval.embedding_time_ms
    # The composed memory is discarded after generation.
    assert namespace_stats(engine._namespace)["num_users"] == 0

    empty = engine.answer("Anything?", [])
    assert engine._encoder.composed[-1] == ["scaffold:header", "scaffold:empty", "scaffold:footer"]
    assert empty.metrics["retrieved_chunk_count"] == 0

    engine.close()
    engine.close()
    assert fake_engine.closed and engine._encoder is None and engine._index.closed
    with pytest.raises(RuntimeError, match="closed"):
        engine.query("again")


def test_failed_gpu_lookup_setup_prevents_serving(engine, monkeypatch) -> None:
    engine.precompute(_chunks())

    def fail_lookup() -> None:
        raise RuntimeError("GPU chunk map allocation failed.")

    monkeypatch.setattr(engine._corpus.store, "finalize_chunk_lookup", fail_lookup)
    with pytest.raises(RuntimeError, match="GPU chunk map allocation failed"):
        engine.start()
    assert engine._engine.started is False
    with pytest.raises(RuntimeError, match="start\\(\\)"):
        engine.query("Who has a cat?")
