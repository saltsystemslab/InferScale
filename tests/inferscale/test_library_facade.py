from __future__ import annotations

import pytest

import inferscale.v1.api as api_module
from inferscale.v1 import Chunk, InferScale, InferScaleConfig
from inferscale.v1.kv.registry import namespace_stats
from library_fakes import FakeChunkStore, FakeEmbedder, FakeEncoder, FakeEngine, FakeIndex


class RecordingEncoder(FakeEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plans = []

    def encode_plan(self, plan):
        self.plans.append(plan)
        return super().encode_plan(plan)


@pytest.fixture
def make_engine(monkeypatch):
    monkeypatch.setattr(api_module, "ChunkedRopeEncoder", RecordingEncoder)
    engines = []

    def make(*, context_window=0, max_position=32768, separator="\n"):
        config = InferScaleConfig.from_dict({
            "model": "fake-model", "top_k": 2, "context_window": context_window,
            "engine": {"block_size": 4}, "kv": {"max_position": max_position},
            "prompt": {"chunk_separator": separator},
        })
        facade = InferScale(
            config,
            embedder=FakeEmbedder(),
            index=FakeIndex(),
            chunk_store=FakeChunkStore(),
            engine=FakeEngine(),
        )
        engines.append(facade)
        return facade

    yield make
    for facade in engines:
        facade.close()


@pytest.fixture
def engine(make_engine):
    return make_engine(context_window=2)


def _chunks() -> list[Chunk]:
    return [
        Chunk(id="c1", text="Alice has a cat named Miso.\n"),
        Chunk(id="c2", text="Bob teaches chemistry.\n", token_ids=[70, 71, 72]),
        Chunk(id="c3", text="Alice lives in Berlin.\n"),
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
    assert engine.get_chunk("c3").context_prefix_tokens == len(engine.get_chunk("c1").token_ids) + 3
    assert engine.get_chunk("c3").context_ids == ("c1", "c2")
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


@pytest.mark.parametrize("context_window", [0, 1, 2, 10, 10**30])
def test_windows_use_only_preceding_target_tokens(make_engine, context_window):
    engine = make_engine(context_window=context_window)
    chunks = [
        Chunk("c1", "first", [10, 11]),
        Chunk("c2", "second", [20]),
        Chunk("c3", "third", [30, 31]),
        Chunk("c4", "fourth", [40]),
    ]
    stats = engine.precompute(chunks)

    assert stats.token_count == 6
    assert engine._embedder.document_calls == [[chunk.text for chunk in chunks]]
    for index, plan in enumerate(engine._encoder.plans):
        previous = chunks[max(0, index - context_window):index] if context_window else []
        prefix = [token for chunk in previous for token in chunk.token_ids]
        target = chunks[index]
        assert plan.context_ids == tuple(chunk.id for chunk in previous)
        assert plan.context_token_ids == prefix
        assert plan.input_token_ids == prefix + target.token_ids
        assert (plan.slice_start, plan.slice_end) == (len(prefix), len(prefix) + len(target.token_ids))
        stored = engine._corpus.get(target.id)
        assert stored.token_ids == target.token_ids
        assert all(tensor.shape[1] == len(target.token_ids) for tensor in stored.kv_by_layer.values())


@pytest.mark.parametrize("batch_sizes", [(4,), (1, 1, 1, 1), (1, 0, 2, 1), (2, 2)])
def test_windows_are_identical_across_precompute_batches(make_engine, batch_sizes):
    chunks = [Chunk(f"c{i}", str(i), [i, i + 10]) for i in range(4)]
    single = make_engine(context_window=2)
    single.precompute(chunks)
    batched = make_engine(context_window=2)
    offset = 0
    for size in batch_sizes:
        batched.precompute(iter(chunks[offset:offset + size]))
        offset += size

    assert batched._encoder.plans == single._encoder.plans
    assert [batched.get_chunk(chunk.id) for chunk in chunks] == [single.get_chunk(chunk.id) for chunk in chunks]


def test_window_tokenization_honors_separator_and_supplied_target_tokens(make_engine):
    engine = make_engine(context_window=2, separator=" | ")
    engine.precompute([Chunk("c1", "first"), Chunk("c2", "different text", [8, 9])])
    engine.precompute([Chunk("c3", "third")])

    first, second, third = engine._encoder.plans
    assert first.target_token_ids == engine.tokenizer.encode("first | ", add_special_tokens=False)
    assert second.target_token_ids == [8, 9]
    assert third.context_token_ids == first.target_token_ids + [8, 9]
    assert third.target_token_ids == engine.tokenizer.encode("third | ", add_special_tokens=False)


def test_window_truncates_oldest_tokens_without_truncating_target(make_engine):
    engine = make_engine(context_window=2, max_position=6)
    engine.precompute([
        Chunk("c1", "one", [1, 2, 3]), Chunk("c2", "two", [4, 5, 6]),
        Chunk("c3", "three", [7, 8]), Chunk("c4", "four", [9, 10, 11, 12, 13, 14]),
    ])

    third, fourth = engine._encoder.plans[2:]
    assert third.context_token_ids == [3, 4, 5, 6]
    assert third.input_token_ids == [3, 4, 5, 6, 7, 8]
    stored = engine._corpus.get("c3")
    assert stored.token_ids == [7, 8]
    assert stored.context_ids == ("c1", "c2")
    assert stored.raw_context_prefix_tokens == 6
    assert stored.context_prefix_tokens == 4
    assert stored.context_prefix_truncated_tokens == 2
    assert fourth.context_token_ids == []
    assert fourth.target_token_ids == [9, 10, 11, 12, 13, 14]
    assert fourth.context_ids == ("c2", "c3")


def test_rejected_batch_does_not_enter_later_windows(make_engine):
    engine = make_engine(context_window=2, max_position=4)
    engine.precompute([Chunk("first", "first", [1, 2])])
    with pytest.raises(RuntimeError, match="exceeding max_position"):
        engine.precompute([
            Chunk("uncommitted", "uncommitted", [3]),
            Chunk("too-long", "too long", [4, 5, 6, 7, 8]),
        ])

    assert engine.chunk_ids() == ["first"]
    assert engine._embedder.document_calls == [["first"]]
    engine.precompute([Chunk("next", "next", [9])])
    assert engine._encoder.plans[-1].context_ids == ("first",)
    assert engine._encoder.plans[-1].context_token_ids == [1, 2]


def test_returned_metadata_cannot_change_later_window_tokens(make_engine):
    engine = make_engine(context_window=1)
    tokens = [10, 11]
    engine.precompute([Chunk("first", "first", tokens)])

    engine.get_chunk("first").token_ids[:] = [999]
    tokens[:] = [888]
    engine.precompute([Chunk("next", "next", [20])])

    assert engine._encoder.plans[-1].context_token_ids == [10, 11]
    assert engine.get_chunk("first").token_ids == [10, 11]
    assert engine._corpus.get("first").token_ids == [10, 11]
    assert engine.get_chunk("missing") is None
