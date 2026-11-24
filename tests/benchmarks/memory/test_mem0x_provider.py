from __future__ import annotations

from library_fakes import FakeEmbedder

from benchmarks.common.vector_types import VectorStoreConfig
from benchmarks.memory.mem0.memory_handles import close_memory
from benchmarks.memory.mem0.provider import (
    Mem0Settings,
    create_mem0_memory,
    register_mem0_jasper_provider,
)

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def test_every_vector_store_config_created_after_registration_knows_the_jasper_provider(tmp_path):
    from mem0.vector_stores.configs import VectorStoreConfig as Mem0VectorStoreConfig

    register_mem0_jasper_provider()
    first = Mem0VectorStoreConfig(provider="jasper", config={"path": str(tmp_path)})
    second = Mem0VectorStoreConfig(provider="jasper", config={"path": str(tmp_path)})

    assert first._provider_configs["jasper"] == second._provider_configs["jasper"] == "JasperConfig"
    assert first._provider_configs is not second._provider_configs


def test_every_mem0_memory_embeds_with_one_shared_local_model(tmp_path, monkeypatch):
    from benchmarks.memory.mem0 import local_embedder

    loaded_models = []

    class LoadedModel(FakeEmbedder):
        def __init__(self, *, model, device):
            super().__init__()
            loaded_models.append((model, device))

    monkeypatch.setattr(local_embedder, "LocalEmbedder", LoadedModel)
    monkeypatch.setattr(local_embedder, "_loaded_embedders", {})
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0-home"))
    memories = [
        create_mem0_memory(
            Mem0Settings(
                store_root=tmp_path / store_name,
                vector_config=VectorStoreConfig(backend="qdrant"),
                embedding_model=EMBEDDING_MODEL,
                embedding_device="cpu",
                memory_llm_model="answer/model",
            )
        )
        for store_name in ("first", "second")
    ]

    try:
        first, second = (memory.embedding_model for memory in memories)
        assert loaded_models == [(EMBEDDING_MODEL, "cpu")]
        assert first.embed("a fact", "search") == second.embed_batch(["a fact"], "add")[0]
        assert first.config.embedding_dims == FakeEmbedder.dim
    finally:
        for memory in memories:
            close_memory(memory)
