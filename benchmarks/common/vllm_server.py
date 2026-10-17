"""The ``vllm serve`` command line for the local servers the benchmarks start."""

from __future__ import annotations

from urllib.parse import SplitResult, urlsplit

from benchmarks.common.json_fields import ConfigError

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def require_local_http_endpoint(url: str, *, where: str) -> SplitResult:
    """Parse a base URL that must point at a plain HTTP server on this machine."""
    endpoint = urlsplit(url)
    if endpoint.scheme != "http" or endpoint.hostname not in LOCAL_HOSTS:
        raise ConfigError(
            f"{where} must be a local HTTP endpoint such as http://127.0.0.1:8000/v1."
        )
    return endpoint


def vllm_serve_command(
    model: str,
    endpoint: SplitResult,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int | None = None,
    quantization: str | None = None,
) -> list[str]:
    command = [
        "vllm",
        "serve",
        model,
        "--host",
        str(endpoint.hostname),
        "--port",
        str(endpoint.port or 80),
        "--trust-remote-code",
        "--dtype",
        dtype,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if tensor_parallel_size is not None:
        command += ["--tensor-parallel-size", str(tensor_parallel_size)]
    if quantization:
        command += ["--quantization", quantization]
    return command
