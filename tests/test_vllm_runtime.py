from types import SimpleNamespace

from locomo_jasper_bench.kv.vllm_runtime import (
    build_strict_gpu_kv_transfer_config,
    common_vllm_kwargs,
)


def test_kv_transfer_config_can_enable_prefix_scan_without_default_user() -> None:
    config = build_strict_gpu_kv_transfer_config(
        connector_module="example.connector",
        namespace="throughput",
        default_user_id=None,
        allow_prefix_scan=True,
        log_memory_hits=False,
    )

    extra = config["kv_connector_extra_config"]
    assert extra["memory_namespace"] == "throughput"
    assert extra["allow_prefix_scan"] is True
    assert extra["log_memory_hits"] is False
    assert "default_user_id" not in extra


def test_kv_transfer_config_defaults_to_gpu_store_backend() -> None:
    config = build_strict_gpu_kv_transfer_config(
        connector_module="example.connector",
        namespace="ns",
    )

    extra = config["kv_connector_extra_config"]
    assert extra["memory_store_backend"] == "gpu"
    assert extra["num_staging_slots"] == 4


def test_kv_transfer_config_carries_cpu_pinned_backend() -> None:
    config = build_strict_gpu_kv_transfer_config(
        connector_module="example.connector",
        namespace="ns",
        store_backend="cpu-pinned",
        num_staging_slots=8,
    )

    extra = config["kv_connector_extra_config"]
    assert extra["memory_store_backend"] == "cpu-pinned"
    assert extra["num_staging_slots"] == 8


def _engine_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "model": "test/model",
        "kv_dtype": "bfloat16",
        "kv_gpu_memory_utilization": 0.30,
        "kv_block_size": 16,
        "kv_max_model_len": 32768,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_common_vllm_kwargs_enables_prefix_caching_by_default() -> None:
    kwargs = common_vllm_kwargs(_engine_config())
    assert kwargs["enable_prefix_caching"] is True


def test_common_vllm_kwargs_honors_prefix_caching_opt_out() -> None:
    kwargs = common_vllm_kwargs(_engine_config(kv_enable_prefix_caching=False))
    assert kwargs["enable_prefix_caching"] is False
