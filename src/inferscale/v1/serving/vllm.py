"""In-process vLLM engine: construction kwargs, KV connector wiring, timing."""

from __future__ import annotations

import gc
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..kv.connector_utils import DEFAULT_KV_STAGING_SLOTS, DEFAULT_KV_STORE_BACKEND

if TYPE_CHECKING:
    from ..config import EngineConfig

logger = logging.getLogger(__name__)

# Convenience env vars some shells export that newer vLLM rejects as unknown.
_UNKNOWN_VLLM_ENV = (
    "VLLM_MODEL",
    "VLLM_API_KEY",
    "VLLM_TP",
    "VLLM_GPU_MEMORY_UTILIZATION",
    "VLLM_MAX_MODEL_LEN",
    "VLLM_DTYPE",
    "VLLM_QUANTIZATION",
    "VLLM_BASE_URL",
)

KV_CONNECTOR_NAME = "MemoryKVConnector"


def build_kv_transfer_config(
    *,
    connector_module: str,
    namespace: str,
    default_user_id: str | None = "default",
    allow_prefix_scan: bool = False,
    log_memory_hits: bool = True,
    store_backend: str = DEFAULT_KV_STORE_BACKEND,
    num_staging_slots: int = DEFAULT_KV_STAGING_SLOTS,
) -> dict[str, Any]:
    extra_config: dict[str, Any] = {
        "memory_namespace": namespace,
        "allow_prefix_scan": allow_prefix_scan,
        "log_memory_hits": log_memory_hits,
        "memory_store_backend": store_backend,
        "num_staging_slots": num_staging_slots,
    }
    if default_user_id is not None:
        extra_config["default_user_id"] = default_user_id
    return {
        "kv_connector": KV_CONNECTOR_NAME,
        "kv_role": "kv_both",
        "kv_connector_module_path": connector_module,
        "kv_connector_extra_config": extra_config,
    }


def force_vllm_inprocess_mode() -> None:
    """Force vLLM V1 offline LLM execution into this process.

    Strict GPU KV mode requires the engine in-process; prompt-prefix
    baselines match so both paths measure the same engine execution mode.
    """
    clear_unknown_vllm_env()
    current = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if _is_truthy(current):
        logger.warning(
            "Overriding VLLM_ENABLE_V1_MULTIPROCESSING=%s because KV injection requires "
            "vLLM's offline engine to share this process.",
            current,
        )
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    loaded_envs = sys.modules.get("vllm.envs")
    if loaded_envs is not None and hasattr(loaded_envs, "VLLM_ENABLE_V1_MULTIPROCESSING"):
        setattr(loaded_envs, "VLLM_ENABLE_V1_MULTIPROCESSING", False)


def clear_unknown_vllm_env() -> None:
    """Clear convenience env vars that vLLM treats as unknown."""
    for name in _UNKNOWN_VLLM_ENV:
        os.environ.pop(name, None)


def vllm_engine_kwargs(*, model: str, dtype: str, engine: "EngineConfig") -> dict[str, Any]:
    return {
        "model": model,
        "dtype": dtype,
        "trust_remote_code": True,
        "enable_prefix_caching": bool(engine.enable_prefix_caching),
        "disable_log_stats": False,
        "swap_space": 0,
        "cpu_offload_gb": 0,
        "gpu_memory_utilization": engine.gpu_memory_utilization,
        "block_size": engine.block_size,
        "max_model_len": engine.max_model_len,
    }


def empty_cuda_cache(*, collect_ipc: bool = False) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        if collect_ipc:
            torch.cuda.ipc_collect()
    except Exception:
        return


@dataclass(frozen=True)
class VllmRequestTiming:
    time_to_first_token_ms: float | None = None


def require_engine_ttft_ms(output: Any) -> float:
    """Return the engine-reported TTFT, refusing to run without real metrics.

    TTFT is a headline metric; fabricating it from wall-clock probes or
    reporting it as missing would silently corrupt cross-run comparisons, so
    an engine build that does not populate RequestOutput.metrics fails.
    """
    timing = request_timing_from_output(output)
    if timing.time_to_first_token_ms is None:
        raise RuntimeError(
            "vLLM did not report per-request metrics on RequestOutput.metrics; "
            "TTFT cannot be measured on this engine build. Use a vLLM build that "
            "populates request timing metrics."
        )
    return timing.time_to_first_token_ms


def request_timing_from_output(output: Any) -> VllmRequestTiming:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return VllmRequestTiming()

    first_token_latency = _float_attr(metrics, "first_token_latency")
    if first_token_latency is not None and first_token_latency > 0:
        return VllmRequestTiming(time_to_first_token_ms=first_token_latency * 1000)

    arrival_time = _float_attr(metrics, "arrival_time")
    first_token_time = _float_attr(metrics, "first_token_time")
    if arrival_time is not None and first_token_time is not None and first_token_time >= arrival_time:
        return VllmRequestTiming(time_to_first_token_ms=(first_token_time - arrival_time) * 1000)

    return VllmRequestTiming()


@dataclass(slots=True)
class GenerationOutput:
    text: str
    ttft_ms: float
    started_at: float
    finished_at: float
    raw: Any


class VLLMEngine:
    """One in-process vLLM engine, optionally with the memory KV connector."""

    def __init__(self, model: str, dtype: str, config: "EngineConfig") -> None:
        self._model = model
        self._dtype = dtype
        self._config = config
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_cls: Any | None = None

    @property
    def started(self) -> bool:
        return self._llm is not None

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("VLLMEngine.start() must be called before using the tokenizer.")
        return self._tokenizer

    def start(self, *, kv_transfer_config: Mapping[str, Any] | None = None) -> float:
        """Construct the engine; returns the startup time in seconds."""
        if self._llm is not None:
            return 0.0
        force_vllm_inprocess_mode()
        from vllm import LLM, SamplingParams

        kwargs = vllm_engine_kwargs(model=self._model, dtype=self._dtype, engine=self._config)
        if kv_transfer_config is not None:
            kwargs["kv_transfer_config"] = dict(kv_transfer_config)
        started = time.perf_counter()
        try:
            self._sampling_cls = SamplingParams
            self._llm = LLM(**kwargs)
            self._tokenizer = self._llm.get_tokenizer()
        except Exception:
            self.close()
            raise
        return time.perf_counter() - started

    def generate(
        self,
        prompt_token_ids: Sequence[int],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> GenerationOutput:
        if self._llm is None or self._sampling_cls is None:
            raise RuntimeError("VLLMEngine.start() must be called before generate().")
        sampling = self._sampling_cls(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        started_at = time.perf_counter()
        outputs = self._llm.generate(
            [{"prompt_token_ids": list(prompt_token_ids)}],
            sampling,
            use_tqdm=False,
        )
        finished_at = time.perf_counter()
        ttft_ms = require_engine_ttft_ms(outputs[0])
        return GenerationOutput(
            text=outputs[0].outputs[0].text.strip(),
            ttft_ms=ttft_ms,
            started_at=started_at,
            finished_at=finished_at,
            raw=outputs[0],
        )

    def close(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._tokenizer = None
        self._sampling_cls = None
        empty_cuda_cache(collect_ipc=True)


def _float_attr(value: Any, name: str) -> float | None:
    raw = getattr(value, name, None)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _is_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}
