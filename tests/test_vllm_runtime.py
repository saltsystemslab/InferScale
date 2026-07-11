from locomo_jasper_bench.kv.vllm_runtime import build_strict_gpu_kv_transfer_config


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
