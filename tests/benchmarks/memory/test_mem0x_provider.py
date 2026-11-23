from __future__ import annotations

from benchmarks.memory.mem0.provider import register_mem0_jasper_provider


def test_every_vector_store_config_created_after_registration_knows_the_jasper_provider(tmp_path):
    from mem0.vector_stores.configs import VectorStoreConfig as Mem0VectorStoreConfig

    register_mem0_jasper_provider()
    first = Mem0VectorStoreConfig(provider="jasper", config={"path": str(tmp_path)})
    second = Mem0VectorStoreConfig(provider="jasper", config={"path": str(tmp_path)})

    assert first._provider_configs["jasper"] == second._provider_configs["jasper"] == "JasperConfig"
    assert first._provider_configs is not second._provider_configs
