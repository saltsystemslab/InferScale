from __future__ import annotations

from typing import Any

import pytest
from library_fakes import FakeChunkStore, FakeEmbedder, FakeEncoder, FakeEngine, FakeIndex

import inferscale.v1.api as api_module
from inferscale.v1 import Chunk, InferScale, InferScaleConfig
from inferscale.v1.types import SearchHit

SMALL_CONFIG = {"model": "m", "top_k": 2, "engine": {"block_size": 4}}


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "ChunkedRopeEncoder", FakeEncoder)


def fake_collaborators() -> dict[str, Any]:
    return {
        "embedder": FakeEmbedder(),
        "index": FakeIndex(),
        "chunk_store": FakeChunkStore(),
        "engine": FakeEngine(),
    }


def facade_with(config: InferScaleConfig | None = None, **collaborators: Any) -> InferScale:
    return InferScale(
        config or InferScaleConfig.from_dict(SMALL_CONFIG),
        **{**fake_collaborators(), **collaborators},
    )


def test_context_window_counts_the_preceding_chunks_used_only_while_encoding() -> None:
    config = InferScaleConfig.from_dict({**SMALL_CONFIG, "context_window": 2})
    chunks = [
        Chunk("c1", "one", [1, 2]),
        Chunk("c2", "two", [3]),
        Chunk("c3", "three", [4, 5]),
        Chunk("c4", "four", [6]),
    ]

    with facade_with(config) as facade:
        facade.precompute(chunks)
        facade.start()

        encoded_after = [facade.get_chunk(chunk.id).context_ids for chunk in chunks]
        assert encoded_after == [(), ("c1",), ("c1", "c2"), ("c2", "c3")]
        assert facade.get_chunk("c4").context_prefix_tokens == len([3, 4, 5])

        scaffold = facade._corpus.scaffold
        answer = facade.answer("Which chunk?", [SearchHit("c4", {}, 1.0, -1.0, 1)])
        own_tokens_of_c4 = 1
        assert answer.metrics["kv_memory_tokens"] == (
            len(scaffold.header.token_ids) + own_tokens_of_c4 + len(scaffold.footer.token_ids)
        )
