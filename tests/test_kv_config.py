from __future__ import annotations

from locomo_jasper_bench.config import parse_args
from locomo_jasper_bench.kv.answer_client import build_strict_gpu_kv_transfer_config


def test_parse_vllm_kv_args() -> None:
    config = parse_args(
        [
            "--answer-backend",
            "vllm-kv",
            "--kv-connector-module",
            "locomo_jasper_bench.kv.strict_gpu_connector",
            "--kv-gpu-memory-utilization",
            "0.42",
            "--kv-max-model-len",
            "8192",
            "--kv-max-position",
            "8192",
            "--kv-dtype",
            "float16",
            "--kv-device",
            "cuda:1",
        ]
    )

    assert config.answer_backend == "vllm-kv"
    assert config.kv_connector_module == "locomo_jasper_bench.kv.strict_gpu_connector"
    assert config.kv_gpu_memory_utilization == 0.42
    assert config.kv_max_model_len == 8192
    assert config.kv_max_position == 8192
    assert config.kv_dtype == "float16"
    assert config.kv_device == "cuda:1"


def test_strict_gpu_transfer_config_has_no_memory_path() -> None:
    transfer = build_strict_gpu_kv_transfer_config(
        connector_module="locomo_jasper_bench.kv.strict_gpu_connector",
        namespace="run-1",
    )

    assert transfer["kv_connector"] == "MemoryKVConnector"
    assert transfer["kv_role"] == "kv_both"
    assert transfer["kv_connector_module_path"] == "locomo_jasper_bench.kv.strict_gpu_connector"
    assert transfer["kv_connector_extra_config"] == {"memory_namespace": "run-1"}
    assert "memory_path" not in transfer["kv_connector_extra_config"]

