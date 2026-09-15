from inferscale.v1.config import EngineConfig
from inferscale.v1.serving.vllm import build_kv_transfer_config, vllm_engine_kwargs


def test_kv_transfer_config_can_enable_prefix_scan_without_default_user() -> None:
    config = build_kv_transfer_config(
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
    config = build_kv_transfer_config(
        connector_module="example.connector",
        namespace="ns",
    )

    extra = config["kv_connector_extra_config"]
    assert extra["memory_store_backend"] == "gpu"
    assert extra["num_staging_slots"] == 4


def test_kv_transfer_config_carries_cpu_pinned_backend() -> None:
    config = build_kv_transfer_config(
        connector_module="example.connector",
        namespace="ns",
        store_backend="cpu",
        num_staging_slots=8,
    )

    extra = config["kv_connector_extra_config"]
    assert extra["memory_store_backend"] == "cpu"
    assert extra["num_staging_slots"] == 8


def test_engine_kwargs_enables_prefix_caching_by_default() -> None:
    kwargs = vllm_engine_kwargs(model="test/model", dtype="bfloat16", engine=EngineConfig())
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["model"] == "test/model"
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["block_size"] == 16
    assert kwargs["max_model_len"] == 32768
    assert kwargs["swap_space"] == 0 and kwargs["cpu_offload_gb"] == 0
    assert kwargs["trust_remote_code"] is True


def test_engine_kwargs_honors_prefix_caching_opt_out() -> None:
    kwargs = vllm_engine_kwargs(
        model="test/model", dtype="bfloat16", engine=EngineConfig(enable_prefix_caching=False)
    )
    assert kwargs["enable_prefix_caching"] is False
