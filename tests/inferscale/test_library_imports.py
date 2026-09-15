from __future__ import annotations

import importlib
import subprocess
import sys


def test_public_api_imports_without_gpu_stacks() -> None:
    """The v1 package must import with torch, vllm, jasper, and openai unavailable."""
    code = (
        "import sys\n"
        "for name in ('torch', 'vllm', 'jasper', 'openai', 'transformers'):\n"
        "    sys.modules[name] = None\n"
        "import inferscale\n"
        "from inferscale.v1 import InferScale, InferScaleConfig, Chunk, KVChunk, Retriever, JasperIndex, MatmulIndex\n"
        "print(len(inferscale.__all__))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert int(result.stdout.strip()) > 20


def test_registry_and_connector_have_single_module_paths() -> None:
    registry = importlib.import_module("inferscale.v1.kv.registry")
    assert registry.__name__ == "inferscale.v1.kv.registry"
    from inferscale.v1.kv import registry as registry_again

    assert registry_again is registry
