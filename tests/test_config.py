from __future__ import annotations

import pytest

from locomo_jasper_bench.config import DEFAULT_MODEL, BenchmarkConfig, parse_args


def test_benchmark_config_defaults_to_gemma_without_judge_fields() -> None:
    config = BenchmarkConfig()

    assert DEFAULT_MODEL == "google/gemma-3-12b-it"
    assert config.model == DEFAULT_MODEL
    data = config.to_jsonable()
    assert data["model"] == DEFAULT_MODEL
    assert "judge_model" not in data
    assert "judge_base_url" not in data
    assert "judge_api_key" not in data


def test_parse_args_rejects_judge_specific_flags() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--judge-model", "google/gemma-3-12b-it"])
