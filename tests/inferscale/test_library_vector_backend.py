from __future__ import annotations

import sys

import pytest

import inferscale.v1.api as api_module
from inferscale.v1 import Chunk, InferScale, InferScaleConfig, JasperIndex, MatmulIndex
from library_fakes import FakeChunkStore, FakeEmbedder, FakeEncoder, FakeEngine, FakeIndex


def test_facade_defaults_to_exact_without_loading_native_stacks(monkeypatch) -> None:
    for module in ("torch", "jasper", "tvm_ffi", "vllm"):
        monkeypatch.setitem(sys.modules, module, None)
    config = InferScaleConfig(model="m")
    assert config.vector_backend == "exact"
    with InferScale(
        config, embedder=FakeEmbedder(), chunk_store=FakeChunkStore(), engine=FakeEngine()
    ) as facade:
        assert isinstance(facade._index, MatmulIndex)


@pytest.mark.parametrize("backend, index_class", [("jasper", JasperIndex), ("exact", MatmulIndex)])
def test_facade_selects_configured_index_without_loading_native_stacks(
    monkeypatch, backend, index_class
) -> None:
    for module in ("torch", "jasper", "tvm_ffi"):
        monkeypatch.setitem(sys.modules, module, None)
    config = InferScaleConfig(model="m", vector_backend=backend)
    with InferScale(
        config, embedder=FakeEmbedder(), chunk_store=FakeChunkStore(), engine=FakeEngine()
    ) as facade:
        assert isinstance(facade._index, index_class)


@pytest.mark.parametrize("backend", ["jasper", "exact"])
def test_explicit_index_takes_precedence_over_backend(monkeypatch, backend) -> None:
    def unexpected_constructor(*args, **kwargs):
        pytest.fail("A supplied index must bypass the configured index constructor.")

    monkeypatch.setattr(api_module, "JasperIndex", unexpected_constructor)
    monkeypatch.setattr(api_module, "MatmulIndex", unexpected_constructor)
    index = FakeIndex()
    with InferScale(
        InferScaleConfig(model="m", vector_backend=backend),
        embedder=FakeEmbedder(), index=index, chunk_store=FakeChunkStore(), engine=FakeEngine(),
    ) as facade:
        assert facade._index is index


def test_matmul_receives_configured_cuda_device(monkeypatch) -> None:
    devices = []

    def make_index(*, device):
        devices.append(device)
        return FakeIndex()

    monkeypatch.setattr(api_module, "MatmulIndex", make_index)
    config = InferScaleConfig.from_dict({
        "model": "m", "vector_backend": "exact", "kv": {"device": "cuda:2"},
    })
    with InferScale(
        config, embedder=FakeEmbedder(), chunk_store=FakeChunkStore(), engine=FakeEngine()
    ):
        assert devices == ["cuda:2"]


def test_matmul_precompute_start_query_uses_chunk_lookup_without_jasper(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setitem(sys.modules, "jasper", None)
    monkeypatch.setitem(sys.modules, "tvm_ffi", None)
    monkeypatch.setattr(api_module, "ChunkedRopeEncoder", FakeEncoder)
    store = FakeChunkStore()
    config = InferScaleConfig.from_dict({
        "model": "m", "top_k": 2, "vector_backend": "exact", "engine": {"block_size": 4},
    })
    with InferScale(
        config, embedder=FakeEmbedder(), chunk_store=store, engine=FakeEngine()
    ) as facade:
        facade.precompute([
            Chunk(id="c1", text="Alice lives in Berlin."),
            Chunk(id="c2", text="Alice enjoys hiking."),
            Chunk(id="c3", text="Bob teaches chemistry."),
            Chunk(id="c4", text="Alice has a cat named Miso."),
        ])
        facade.start()
        result = facade.query("What is Alice's cat named?")
        assert len(result.hits) == 2
        assert result.retrieval.vector_backend == "exact"
        assert result.retrieval.jasper_effective_beam_width is None
        assert store.lookup_batches == [[hit.id for hit in reversed(result.hits)]]
