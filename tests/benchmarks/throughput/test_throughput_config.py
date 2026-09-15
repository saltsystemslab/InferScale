from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig, load_json_object
from benchmarks.common.paths import project_root
from benchmarks.memory.config import MemoryRunConfig
from benchmarks.memory.throughput.config import (
    ALL_CONDITIONS,
    DEFAULT_USER_COUNTS,
    ThroughputConfig,
    condition_vector_backend,
    load_throughput_config,
)


@pytest.fixture
def runtime(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig.from_dict(
        {
            "models": {"llama": "local/llama-test"},
            "storage": {"runtime_root": str(tmp_path / "persistent"), "local_store_dir": str(tmp_path / "scratch")},
        },
        root=tmp_path,
    )


def _write(tmp_path: Path, data: dict[str, Any], name: str = "run.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _data(**values: Any) -> dict[str, Any]:
    return {"benchmark": "memory-throughput", "model": "test/model", "inferscale": {}, **values}


def _config(tmp_path: Path, runtime: RuntimeConfig, **values: Any) -> ThroughputConfig:
    return load_throughput_config(_write(tmp_path, _data(**values)), runtime)


def test_main_resolves_json_model_alias_paths_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_path = _write(
        tmp_path,
        {"models": {"llama": "local/llama-test"}, "storage": {"runtime_root": str(tmp_path)}},
        "runtime.json",
    )
    path = _write(tmp_path, _data(
        model="llama", conditions=["mem0_jasper", "mem0_qdrant"], user_counts=[2, 3],
        dataset_path="data/other.json", run_id="unit-test",
    ))
    applied: list[Path] = []
    monkeypatch.setattr(RuntimeConfig, "apply_environment", lambda self: applied.append(self.path))
    from benchmarks.memory.throughput.run import main

    runs = []
    monkeypatch.setattr(
        "benchmarks.memory.throughput.runner.run_throughput",
        lambda config, *, dry_run: runs.append((config, dry_run)),
    )
    main(path, runtime_path=runtime_path, dry_run=True)
    config, dry_run = runs[0]

    assert config.model == "local/llama-test"
    assert config.model_label == "llama"
    assert config.user_counts == (2, 3)
    assert config.dataset_path == project_root() / "data/other.json"
    assert config.results_dir == tmp_path / "results"
    assert config.conditions == ("mem0_jasper", "mem0_qdrant")
    assert dry_run is True
    assert applied == [runtime_path]


def test_runtime_storage_defaults_survive_worker_snapshot(tmp_path: Path, runtime: RuntimeConfig) -> None:
    config = _config(tmp_path, runtime, run_id="unit-test")

    assert config.results_dir == tmp_path / "persistent/results"
    assert config.embedding_cache_dir == tmp_path / "persistent/.cache/embeddings"
    assert config.memory_llm_cache_dir == tmp_path / "persistent/.cache/mem0-inference"
    assert config.local_store_scratch_dir == tmp_path / "scratch/inferscale-bench-stores/unit-test"
    restored = ThroughputConfig.from_json_file(_write(tmp_path, config.to_jsonable(), "snapshot.json"))
    assert restored.to_jsonable() == config.to_jsonable()


def test_worker_snapshot_without_local_store_dir_uses_runtime_default(tmp_path: Path) -> None:
    config = ThroughputConfig(model="test/model", model_label="test", results_dir=tmp_path, run_id="legacy")
    payload = config.to_jsonable()
    payload.pop("local_store_dir")

    assert ThroughputConfig.from_json_file(_write(tmp_path, payload)).local_store_dir == config.local_store_dir


def test_default_conditions_and_values(tmp_path: Path, runtime: RuntimeConfig) -> None:
    config = _config(tmp_path, runtime)

    assert config.conditions == ALL_CONDITIONS == ("mem0_qdrant", "mem0_jasper", "kv_injection")
    assert config.user_counts == DEFAULT_USER_COUNTS
    assert config.memory_llm_provider == "vllm"
    assert config.memory_llm_model == config.model
    assert config.top_k == config.context_window == 50
    assert config.kv_store_backend == "gpu"
    assert config.kv_staging_slots == 4
    assert config.kv_enable_prefix_caching is True
    assert "vector_distance" not in config.to_jsonable()


@pytest.mark.parametrize("mem0", [{}, {"llm_base_url": None}])
def test_fresh_throughput_and_memory_configs_share_extraction_endpoint(
    tmp_path: Path, runtime: RuntimeConfig, mem0: dict[str, Any]
) -> None:
    config = _config(tmp_path, runtime, mem0=mem0)
    direct = ThroughputConfig(model="test/model", model_label="test", results_dir=tmp_path, run_id="endpoint")
    memory_data = load_json_object(project_root() / "configs/memory/accuracy-latency/llama.json")
    memory_data["mem0"] = mem0
    memory = MemoryRunConfig.from_dict(memory_data, runtime=runtime)
    assert config.memory_llm_base_url == direct.memory_llm_base_url == memory.memory_llm_base_url
    assert config.memory_llm_base_url == "http://localhost:8000/v1"


def test_json_controls_settings_and_environment_only_supplies_credentials(
    tmp_path: Path, runtime: RuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in {
        "LOCOMO_VLLM_MODEL": "other/model", "THROUGHPUT_USER_COUNTS": "999", "THROUGHPUT_TOP_K": "1",
        "LOCOMO_KV_STORE_BACKEND": "cpu", "LOCOMO_KV_ENABLE_PREFIX_CACHING": "0",
        "LOCOMO_LOG_LEVEL": "DEBUG", "OPENAI_EMBEDDING_MODEL": "other-embedding",
        "OPENAI_BASE_URL": "https://environment.example/v1", "EXTRACTION_LLM_BASE_URL": "https://environment.example/v1",
        "LOCOMO_JASPER_DEVICE_KV_SELECTION": "0", "OPENAI_API_KEY": "secret-value",
    }.items():
        monkeypatch.setenv(key, value)
    config = _config(tmp_path, runtime, model="llama", user_counts=[2, 3], log_level="WARNING", inferscale={
        "embedding": {"model": "json-embedding", "base_url": "https://embedding.example/v1"},
    }, mem0={"llm_base_url": "https://extraction.example/v1"})

    assert config.model == "local/llama-test"
    assert config.user_counts == (2, 3)
    assert config.top_k == 50
    assert config.kv_store_backend == "gpu"
    assert config.kv_enable_prefix_caching is True
    assert config.log_level == "WARNING"
    assert config.embedding_model == "json-embedding"
    assert config.embedding_base_url == "https://embedding.example/v1"
    assert config.memory_llm_base_url == "https://extraction.example/v1"
    assert config.embedding_api_key == "secret-value"
    assert not hasattr(config, "jasper_device_kv_selection")
    assert "secret-value" not in json.dumps(config.to_jsonable())


@pytest.mark.parametrize("endpoint", [None, "http://localhost:9000/v1"])
def test_extraction_endpoint_round_trips_without_environment_override(
    tmp_path: Path, runtime: RuntimeConfig, monkeypatch: pytest.MonkeyPatch, endpoint: str | None
) -> None:
    monkeypatch.setenv("EXTRACTION_LLM_BASE_URL", "https://ignored.example/v1")
    config = _config(tmp_path, runtime, mem0={"llm_base_url": endpoint})
    restored = ThroughputConfig.from_json_file(_write(tmp_path, config.to_jsonable(), "snapshot.json"))
    expected = endpoint if endpoint is not None else "http://localhost:8000/v1"
    assert config.memory_llm_base_url == restored.memory_llm_base_url == expected


def test_legacy_worker_snapshot_preserves_explicit_null_extraction_endpoint(
    tmp_path: Path, runtime: RuntimeConfig
) -> None:
    payload = _config(tmp_path, runtime).to_jsonable()
    payload["memory_llm_base_url"] = None
    restored = ThroughputConfig.from_json_file(_write(tmp_path, payload, "snapshot.json"))

    assert restored.memory_llm_base_url is None
    assert restored.to_jsonable()["memory_llm_base_url"] is None


def test_memory_llm_model_always_matches_the_answer_model(tmp_path: Path, runtime: RuntimeConfig) -> None:
    config = _config(tmp_path, runtime)
    payload = config.to_jsonable()
    payload["memory_llm_model"] = "other/model"
    with pytest.raises(ValueError, match="always uses the answer model"):
        ThroughputConfig.from_json_file(_write(tmp_path, payload, "snapshot.json"))


def test_condition_vector_backends_pair_kv_with_jasper() -> None:
    assert condition_vector_backend("mem0_qdrant") == "qdrant"
    assert condition_vector_backend("mem0_jasper") == "jasper"
    assert condition_vector_backend("kv_injection") == "jasper"


def test_worker_config_restores_redacted_api_key_from_environment(
    tmp_path: Path, runtime: RuntimeConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    config = _config(tmp_path, runtime, conditions=["mem0_qdrant"], user_counts=[2])
    path = _write(tmp_path, config.to_jsonable(), "snapshot.json")
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("LOCOMO_THROUGHPUT_EMBEDDING_API_KEY", "secret-value")

    restored = ThroughputConfig.from_json_file(path)
    assert restored.embedding_api_key == "secret-value"
    assert restored.user_counts == (2,)
    assert restored.dataset_path == config.dataset_path
    assert restored.memory_llm_cache_dir == config.memory_llm_cache_dir
    assert "secret-value" not in path.read_text(encoding="utf-8")


def test_cpu_payload_store_context_and_prefix_cache_round_trip(tmp_path: Path, runtime: RuntimeConfig) -> None:
    config = _config(tmp_path, runtime, context_window=0, inferscale={
        "kv": {"store_backend": "cpu", "staging_slots": 8},
        "engine": {"enable_prefix_caching": False},
    })
    restored = ThroughputConfig.from_json_file(_write(tmp_path, config.to_jsonable(), "snapshot.json"))

    assert config.kv_store_backend == restored.kv_store_backend == "cpu"
    assert config.kv_staging_slots == restored.kv_staging_slots == 8
    assert config.kv_enable_prefix_caching is restored.kv_enable_prefix_caching is False
    assert config.context_window == restored.context_window == 0


@pytest.mark.parametrize("value", [False, True])
def test_worker_snapshot_discards_removed_device_kv_selection_switch(
    tmp_path: Path, runtime: RuntimeConfig, value: bool
) -> None:
    config = _config(tmp_path, runtime)
    payload = config.to_jsonable()
    payload["jasper_device_kv_selection"] = value
    restored = ThroughputConfig.from_json_file(_write(tmp_path, payload, "snapshot.json"))
    assert not hasattr(restored, "jasper_device_kv_selection")
    assert restored.to_jsonable() == config.to_jsonable()


@pytest.mark.parametrize("values,match", [
    ({"benchmark": "rag"}, "Expected benchmark"),
    ({"model": " "}, "model"),
    ({"model": 12}, "model"),
    ({"conditions": ["prompt_injection"]}, "conditions"),
    ({"conditions": ["kv_injection", "kv_injection"]}, "duplicates"),
    ({"user_counts": [True]}, "integers"),
    ({"user_counts": [1.5]}, "integers"),
    ({"user_counts": [0]}, "positive"),
    ({"user_counts": [1, 1]}, "duplicates"),
    ({"user_counts": "1,2"}, "integers"),
    ({"requests_per_user": 0}, "requests_per_user"),
    ({"max_output_tokens": 0}, "max_output_tokens"),
    ({"warmup_batches": -1}, "warmup_batches"),
    ({"top_k": 0}, "top_k"),
    ({"context_window": -1}, "context_window"),
    ({"context_window": True}, "context_window"),
    ({"seed": "42"}, "seed"),
    ({"log_level": "invalid"}, "log_level"),
    ({"embedding_cache_enabled": "false"}, "embedding_cache_enabled"),
    ({"inferscale": {"kv": {"staging_slots": 0}}}, "staging_slots"),
    ({"inferscale": {"kv": {"staging_slots": True}}}, "staging_slots"),
    ({"inferscale": {"kv": {"device": 0}}}, "device"),
    ({"inferscale": {"engine": {"enable_prefix_caching": "false"}}}, "enable_prefix_caching"),
    ({"inferscale": {"engine": {"gpu_memory_utilization": 1}}}, "gpu_memory_utilization"),
    ({"inferscale": {"index": {"alpha": -1}}}, "alpha"),
    ({"inferscale": {"index": {"beam_width": 960}}}, "beam width"),
    ({"inferscale": {"index": {"beam_witdh": 64}}}, "unknown keys"),
    ({"inferscale": {"prompt": {}}}, "prompt"),
    ({"inferscale": {"embedding": {"batch_size": 128}}}, "batch_size"),
    ({"inferscale": {"generation": {"max_tokens": 51}}}, "max_tokens"),
    ({"inferscale": {"generation": {"temperature": 1.0}}}, "temperature"),
    ({"inferscale": {"generation": {"top_p": 0.9}}}, "top_p"),
    ({"mem0": {"llm_model": "other/model"}}, "unknown keys"),
    ({"sweeps": {"standard": {"kv_staging_slots": 0}}}, "kv_staging_slots"),
    ({"sweeps": {"standard": {"unknown": True}}}, "unknown keys"),
    ({"sweeps": {"standard": {"conditions": ["unknown"]}}}, "conditions"),
    ({"sweeps": {"standard": {"kv_store_backend": "disk"}}}, "kv_store_backend"),
    ({"kv_device": "cuda:0"}, "unknown keys"),
    ({"vector_distance": "ip"}, "unknown keys"),
    ({"jasper_device_kv_selection": False}, "unknown keys"),
])
def test_authored_json_rejects_invalid_settings(
    tmp_path: Path, runtime: RuntimeConfig, values: dict[str, Any], match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        _config(tmp_path, runtime, **values)


@pytest.mark.parametrize("values,match", [
    ({"user_counts": [1.5]}, "integers"),
    ({"requests_per_user": "2"}, "requests_per_user"),
    ({"vector_distance": "ip"}, "unknown keys"),
    ({"kv_enable_prefix_caching": "false"}, "enable_prefix_caching"),
])
def test_worker_snapshots_reject_invalid_settings(
    tmp_path: Path, runtime: RuntimeConfig, values: dict[str, Any], match: str
) -> None:
    payload = _config(tmp_path, runtime).to_jsonable()
    payload.update(values)
    with pytest.raises(ConfigError, match=match):
        ThroughputConfig.from_json_file(_write(tmp_path, payload, "snapshot.json"))


def test_committed_throughput_presets_load(runtime: RuntimeConfig) -> None:
    for path in sorted((project_root() / "configs/memory/throughput").glob("*.json")):
        config = load_throughput_config(path, runtime)
        assert config.model
        assert config.user_counts
