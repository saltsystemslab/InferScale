from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.common.config import ConfigError, load_runtime_config
from benchmarks.memory.config import load_memory_config
from memory_config import memory_config_data


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in ("JUDGE_LLM_BASE_URL", "JUDGE_LLM_API_KEY", "EXTRACTION_LLM_BASE_URL", "EXTRACTION_LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config_paths(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "memory.json"
    config_path.write_text(
        json.dumps(memory_config_data(model="custom", top_k=7, run_id="json-test")), encoding="utf-8"
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps({
            "models": {"custom": "example-org/custom-model"},
            "storage": {"runtime_root": str(tmp_path), "local_store_dir": str(tmp_path / "local")},
        }),
        encoding="utf-8",
    )
    return config_path, runtime_path


def load(paths: tuple[Path, Path], *, stage: str = "run"):
    config_path, runtime_path = paths
    return load_memory_config(config_path, load_runtime_config(runtime_path), stage=stage)


def test_loader_loads_json_and_runtime_paths(config_paths: tuple[Path, Path], tmp_path: Path) -> None:
    config = load(config_paths)

    assert config.model == "example-org/custom-model"
    assert config.top_k == 7
    assert config.run_dir == tmp_path / "results" / "json-test"
    assert config.memory_llm_cache_dir == tmp_path / ".cache" / "mem0-inference"
    assert config.local_store_scratch_dir() == tmp_path / "local" / "inferscale-bench-stores" / "json-test"


def write_settings(config_paths: tuple[Path, Path], **values) -> None:
    path = config_paths[0]
    data = json.loads(path.read_text())
    data.update(values)
    path.write_text(json.dumps(data))


def test_loader_uses_nested_json_settings(config_paths: tuple[Path, Path]) -> None:
    path = config_paths[0]
    data = json.loads(path.read_text())
    data["top_k"] = 9
    data["inferscale"]["engine"]["block_size"] = 32
    data["judge"]["with_evidence"] = True
    path.write_text(json.dumps(data))
    config = load(config_paths)

    assert config.top_k == 9
    assert config.kv_block_size == 32
    assert config.with_evidence is True
    assert config.kv_max_model_len == 32768


@pytest.mark.parametrize("stage", ["run", "judge", "check-catalogs", "preembed", "precompute-kv"])
def test_loader_accepts_supported_stages(config_paths: tuple[Path, Path], stage: str) -> None:
    assert load(config_paths, stage=stage).run_id == "json-test"


def test_loader_rejects_unknown_stage(config_paths: tuple[Path, Path]) -> None:
    with pytest.raises(ConfigError, match="Unsupported memory stage"):
        load(config_paths, stage="unknown")


def test_loader_rejudge_setting_requires_judge_stage(config_paths: tuple[Path, Path]) -> None:
    write_settings(config_paths, rejudge=True)
    with pytest.raises(ConfigError):
        load(config_paths)
    config = load(config_paths, stage="judge")
    assert config.rejudge is True


@pytest.mark.parametrize("values", [{"skip_judge": True}, {"judge": {"provider": "none"}}])
def test_loader_judge_stage_requires_enabled_judging(config_paths: tuple[Path, Path], values) -> None:
    write_settings(config_paths, **values)
    with pytest.raises(ConfigError):
        load(config_paths, stage="judge")


def test_loader_validates_json_settings(config_paths: tuple[Path, Path]) -> None:
    write_settings(config_paths, context_window=-1)
    with pytest.raises(ConfigError, match="context_window must be >= 0"):
        load(config_paths)


def test_loader_judge_stage_requires_existing_run_id(config_paths: tuple[Path, Path]) -> None:
    write_settings(config_paths, run_id=None)
    with pytest.raises(ConfigError, match="requires run_id in JSON"):
        load(config_paths, stage="judge")


@pytest.mark.parametrize(("name", "value"), [
    ("log_every", -1), ("max_samples", -1), ("max_questions", 0),
    ("run_id", "../other-run"), ("run_id", ""),
])
def test_loader_rejects_invalid_json_bounds_and_run_paths(config_paths, name, value) -> None:
    write_settings(config_paths, **{name: value})
    with pytest.raises(ConfigError, match=name):
        load(config_paths)


def test_loader_role_endpoints_use_json_and_environment_keys_are_redacted(
    config_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    for role in ("JUDGE", "EXTRACTION"):
        monkeypatch.setenv(f"{role}_LLM_BASE_URL", f"https://env-{role.lower()}.example/v1")
        monkeypatch.setenv(f"{role}_LLM_API_KEY", f"env-{role.lower()}-key")
    write_settings(
        config_paths,
        judge={"base_url": "https://json-judge.example/v1"},
        mem0={"llm_base_url": "https://json-extraction.example/v1"},
    )
    config = load(config_paths)

    assert config.judge_base_url == "https://json-judge.example/v1"
    assert config.judge_api_key == "env-judge-key"
    assert config.memory_llm_base_url == "https://json-extraction.example/v1"
    assert config.memory_llm_api_key == "env-extraction-key"
    serialized = config.to_jsonable()
    assert serialized["judge"]["api_key"] == "<redacted>"
    assert serialized["memory_llm_api_key"] == "<redacted>"
    assert "env-judge-key" not in json.dumps(serialized)
    assert "env-extraction-key" not in json.dumps(serialized)
