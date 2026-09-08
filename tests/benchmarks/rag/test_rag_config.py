from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig
from benchmarks.rag.config import RagBenchConfig, load_rag_config


@pytest.fixture
def custom_runtime(tmp_path, monkeypatch) -> RuntimeConfig:
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "storage": {"runtime_root": str(tmp_path)},
                "models": {"llama": "custom/answer-model", "qwen": "custom/qwen"},
                "judge_server": {
                    "model": "custom/judge-model",
                    "base_url": "https://runtime-judge.example/v1",
                    "api_key": "runtime-judge-key",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("JUDGE_LLM_API_KEY", raising=False)
    return RuntimeConfig.from_json(runtime_path, root=tmp_path)


def write_config(tmp_path: Path, data: dict) -> Path:
    target = tmp_path / "rag.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


def test_defaults(custom_runtime) -> None:
    config = RagBenchConfig.from_dict({"skip_judge": True}, runtime=custom_runtime)

    assert config.dataset_name == "multihoprag"
    assert config.data_dir == custom_runtime.root / "data/multihoprag"
    assert config.chunk_size == 1024
    assert config.context_window == 5
    assert config.top_k == 15
    assert config.answer_backend == "kv-injection"
    assert config.result_mode() == "rag-kv"
    assert config.model == "custom/answer-model"
    assert config.max_answer_tokens == 64
    assert config.kv_dtype == "bfloat16"
    assert config.kv_block_size == 16
    assert config.kv_store_backend == "cpu"
    assert config.skip_judge is True and config.judge_provider == "none"


def test_nested_json_populates_downstream_settings(custom_runtime) -> None:
    data = {
        "models": ["qwen"],
        "dataset": "qasper",
        "answer_backend": "prompt-injection",
        "top_k": 3,
        "inferscale": {
            "engine": {"max_model_len": 65536, "gpu_memory_utilization": 0.5},
            "kv": {"max_position": 65536, "device": "cuda:2"},
            "embedding": {"model": "custom/embedding", "base_url": "http://embedding/v1", "batch_size": 20},
            "index": {"beam_width": 32, "n_neighbors": 48},
            "generation": {"max_tokens": 90, "temperature": 0.3, "top_p": 0.9},
        },
    }
    original = deepcopy(data)
    config = RagBenchConfig.from_dict(data, runtime=custom_runtime)

    assert data == original
    assert config.model == "custom/qwen"
    assert config.result_mode() == "rag-prefix"
    assert config.data_dir == custom_runtime.root / "data/qasper"
    assert config.kv_max_model_len == config.kv_max_position == 65536
    assert config.kv_gpu_memory_utilization == 0.5
    assert config.kv_device == "cuda:2"
    assert config.embedding_model == "custom/embedding"
    assert config.embedding_base_url == "http://embedding/v1"
    assert config.embed_batch_size == 20
    assert config.jasper_beam_width == 32
    assert config.jasper_n_neighbors == 48
    assert (config.max_answer_tokens, config.temperature, config.top_p) == (90, 0.3, 0.9)


def test_data_dir_follows_dataset_name() -> None:
    config = RagBenchConfig(dataset_name="qasper")
    assert config.data_dir == Path("data/qasper")


def test_model_alias_resolves_once(tmp_path) -> None:
    runtime = RuntimeConfig.from_dict(
        {"models": {"answer": "actual-model", "actual-model": "unrelated-model"}}, root=tmp_path
    )
    assert RagBenchConfig.from_dict({"model": "answer"}, runtime=runtime).model == "actual-model"


def test_to_jsonable_redacts_secrets_and_stringifies_paths(custom_runtime, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "embedding-secret")
    config = RagBenchConfig.from_dict(
        {"results_dir": "output", "kv_chunk_cache_root": "cache", "run_id": "abc"},
        runtime=custom_runtime,
    )
    data = config.to_jsonable()

    assert data["embedding_api_key"] == "<redacted>"
    assert data["judge_api_key"] == "<redacted>"
    assert data["results_dir"] == str(custom_runtime.root / "output")
    assert data["kv_chunk_cache_root"] == str(custom_runtime.root / "cache")
    assert isinstance(data["data_dir"], str)
    assert data["mode"] == "rag-kv"
    assert data["kv_store_backend"] == "cpu"
    assert data["jasper_effective_beam_width"] == 64
    assert config.run_dir == custom_runtime.root / "output/abc"


@pytest.mark.parametrize(
    "data,match",
    [
        ({"top_k": 0}, "top_k"),
        ({"chunk_size": 0}, "chunk_size"),
        ({"context_window": -1}, "context_window"),
        ({"top_k": True}, "integer"),
        ({"chunk_size": "1024"}, "integer"),
        ({"max_queries": 0}, "max_queries"),
        ({"answer_backend": "other"}, "answer_backend"),
        ({"models": ["llama", "qwen"]}, "one model"),
        ({"models": ["llama"], "model": "llama"}, "not both"),
        ({"top_kk": 5}, "unknown keys"),
        ({"benchmark": "memory"}, "benchmark"),
        ({"dataset": "unknown"}, "Unknown RAG dataset"),
        ({"run_id": "../escape"}, "directory name"),
        ({"inferscale": {"index": {"beam_width": 960}}}, "beam width"),
        ({"inferscale": {"kv": {"store_backend": "gpu"}}}, "must be cpu"),
        ({"inferscale": {"engine": {"typo": 4}}}, "unknown keys"),
        ({"inferscale": {"engine": {"enable_prefix_caching": "false"}}}, "boolean"),
        ({"inferscale": {"index": {"beam_width": True}}}, "integer"),
        ({"sweeps": {"standard": {"top_k": [0]}}}, "top_k"),
    ],
)
def test_invalid_json_settings_rejected(custom_runtime, data, match) -> None:
    with pytest.raises(ConfigError, match=match):
        RagBenchConfig.from_dict(data, runtime=custom_runtime)


def test_memory_budget_validation_rejects_oversized_grids(custom_runtime) -> None:
    with pytest.raises(ConfigError, match="memory budget"):
        RagBenchConfig.from_dict({"top_k": 50}, runtime=custom_runtime)
    config = RagBenchConfig.from_dict(
        {"top_k": 50, "inferscale": {"kv": {"max_position": 65536}, "engine": {"max_model_len": 65536}}},
        runtime=custom_runtime,
    )
    assert config.top_k == 50


def test_prefix_requires_prefix_caching(custom_runtime) -> None:
    with pytest.raises(ConfigError, match="requires.*enable_prefix_caching"):
        RagBenchConfig.from_dict(
            {"answer_backend": "prompt-injection", "inferscale": {"engine": {"enable_prefix_caching": False}}},
            runtime=custom_runtime,
        )


@pytest.mark.parametrize("stage", ["estimate", "preembed", "precompute-kv", "run", "judge"])
def test_config_loader_accepts_supported_stages(tmp_path, custom_runtime, stage) -> None:
    path = write_config(tmp_path, {"max_queries": 25, "run_id": "existing"})
    config = load_rag_config(path, runtime=custom_runtime, stage=stage)
    assert config.max_queries == 25
    assert config.run_id == "existing"


def test_config_loader_rejects_unknown_stage(tmp_path, custom_runtime) -> None:
    path = write_config(tmp_path, {})
    with pytest.raises(ConfigError, match="Unsupported RAG stage"):
        load_rag_config(path, runtime=custom_runtime, stage="unknown")


def test_judge_requires_enabled_provider_and_explicit_run(tmp_path, custom_runtime) -> None:
    for data in ({"skip_judge": True, "run_id": "existing"}, {}, {"rejudge": True}):
        path = write_config(tmp_path, data)
        with pytest.raises(ConfigError, match="judge stage requires"):
            load_rag_config(path, runtime=custom_runtime, stage="judge")
    path = write_config(tmp_path, {"run_id": "existing", "rejudge": True})
    assert load_rag_config(path, runtime=custom_runtime, stage="judge").rejudge
    with pytest.raises(ConfigError, match="rejudge requires the judge stage"):
        load_rag_config(path, runtime=custom_runtime)


def test_explicit_runtime_file_controls_aliases_and_storage(tmp_path, custom_runtime, monkeypatch) -> None:
    other = tmp_path / "other-runtime.json"
    other.write_text(json.dumps({"models": {"llama": "explicit/model"}, "storage": {"runtime_root": str(tmp_path / "other")}}))
    monkeypatch.setenv("INFERSCALE_RUNTIME_CONFIG", "/nonexistent/runtime.json")
    path = write_config(tmp_path, {})
    config = load_rag_config(path, runtime=RuntimeConfig.from_json(other))
    assert config.model == "explicit/model"
    assert config.results_dir == tmp_path / "other/results"


def test_json_settings_ignore_environment_but_credentials_remain(custom_runtime, monkeypatch) -> None:
    for name, value in {
        "JUDGE_LLM_BASE_URL": "http://environment/v1",
        "JUDGE_MODEL": "environment/judge",
        "RAG_MODEL": "environment/model",
        "RAG_TOP_K": "99",
        "RAG_LOG_LEVEL": "ERROR",
        "OPENAI_BASE_URL": "http://environment-embedding/v1",
        "OPENAI_EMBEDDING_MODEL": "environment/embedding",
        "JUDGE_LLM_API_KEY": "environment-judge-key",
        "OPENAI_API_KEY": "environment-embedding-key",
    }.items():
        monkeypatch.setenv(name, value)
    config = RagBenchConfig.from_dict(
        {"judge": {"base_url": "http://json-judge/v1"}, "top_k": 3}, runtime=custom_runtime
    )
    assert config.model == "custom/answer-model"
    assert config.top_k == 3
    assert config.judge_base_url == "http://json-judge/v1"
    assert config.judge_model == "custom/judge-model"
    assert config.judge_api_key == "environment-judge-key"
    assert config.embedding_model == "text-embedding-3-small"
    assert config.embedding_base_url is None
    assert config.embedding_api_key == "environment-embedding-key"
    assert config.log_level == "INFO"


def test_authored_rag_configs_load() -> None:
    for path in Path("configs/rag").glob("*.json"):
        config = RagBenchConfig.from_json_file(path)
        assert config.dataset_name in {"qasper", "multihoprag"}
        assert config.kv_store_backend == "cpu"


def test_chunk_cache_uses_explicit_config_runtime_layout(custom_runtime) -> None:
    from benchmarks.rag.kv_cache import rag_chunk_cache_dir

    config = RagBenchConfig.from_dict({}, runtime=custom_runtime)
    assert config.kv_chunk_cache_root == custom_runtime.layout.rag_kv_chunk_cache_root
    cache_dir = rag_chunk_cache_dir(
        model="custom/answer-model", dtype="bfloat16", chunk_size=1024,
        context_window=5, max_position=32768, corpus_fingerprint="abc123",
        cache_root=config.kv_chunk_cache_root,
    )
    assert cache_dir.is_relative_to(custom_runtime.layout.rag_kv_chunk_cache_root)
