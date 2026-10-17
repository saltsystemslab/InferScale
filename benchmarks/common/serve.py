"""Start the judge server described by the runtime JSON, for scripts/serve_vllm.sh."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from benchmarks.common.config import load_runtime_config
from benchmarks.common.environment import runtime_path_for_config
from benchmarks.common.json_fields import load_json_object, optional, reject_unknown_keys
from benchmarks.common.vllm_server import require_local_http_endpoint, vllm_serve_command


def serve(config_path: Path) -> int:
    """Start, or with ``dry_run`` only print, the judge server command."""
    data = load_json_object(config_path)
    reject_unknown_keys(data, ("runtime_config", "dry_run"), "serve")
    dry_run = optional(data, "dry_run", bool, False, "serve")
    runtime = load_runtime_config(runtime_path_for_config(config_path))
    runtime.apply_environment()
    judge = runtime.judge_server
    endpoint = require_local_http_endpoint(judge.base_url, where="judge_server.base_url")
    command = vllm_serve_command(
        runtime.resolve_model(judge.model),
        endpoint,
        dtype=judge.dtype,
        max_model_len=judge.max_model_len,
        gpu_memory_utilization=judge.gpu_memory_utilization,
        tensor_parallel_size=judge.tensor_parallel_size,
        quantization=judge.quantization,
    )
    print(shlex.join(command), flush=True)
    if dry_run:
        return 0
    runtime.layout.create_directories()
    api_key = runtime.judge_api_key()
    if api_key:
        command += ["--api-key", api_key]
    return subprocess.call(command)
