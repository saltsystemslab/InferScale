"""The LMCache CacheBlend baseline: retrieved passages served from LMCache's segment cache.

Every corpus chunk's KV is stored once in LMCache's pinned-CPU cache when the
engine starts, each chunk alone, the way the KV arm precomputes every chunk.
A query lays the retrieved chunks between separator tokens; LMCache's blender
reuses their KV at the new positions and recomputes a fraction of the tokens,
so vLLM prefills only the footer and the question.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from benchmarks.common.answer_timing import generation_latency_metrics
from benchmarks.common.clients import ChatResult
from benchmarks.common.lmcache.config import engine_config_for_lmcache, lmcache_kv_transfer_config
from benchmarks.common.lmcache.load_stats import reset_load_stats, snapshot_load_stats
from benchmarks.common.lmcache.runtime import (
    GB,
    TAIL_ALLOWANCE_TOKENS,
    kv_bytes_per_token,
    register_model,
    require_cache_capacity,
    require_separator_parity,
    shutdown,
    warm_prompts,
    warmup_metrics,
)
from benchmarks.common.lmcache.segments import (
    blend_memory_layout,
    prompt_token_count,
    warm_up_prompts,
)
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.data_types import RagChunk, RagPromptProfile, RagQuery
from benchmarks.rag.engine_client import RagEngineClient
from inferscale.v1.config import EngineConfig
from inferscale.v1.kv.compose import reverse_ranked_ids
from inferscale.v1.kv.prompt import build_prompt_tokens
from inferscale.v1.serving.vllm import empty_cuda_cache
from inferscale.v1.types import SearchHit

WARM_BATCH_SIZE = 64


class RagLMCacheBlendAnswerClient(RagEngineClient):
    label = "lmcache-blend"

    def __init__(
        self,
        config: RagBenchConfig,
        *,
        chunks: Sequence[RagChunk],
        prompt_profile: RagPromptProfile,
        query_count: int,
    ) -> None:
        super().__init__(
            config,
            chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
            prompt_profile=prompt_profile,
        )
        self._chunks = list(chunks)
        self._query_count = query_count
        self._separator: list[int] | None = None
        self._warm: dict[str, Any] = {}

    def setup_stats(self) -> dict[str, Any]:
        return dict(self._warm)

    def answer(
        self,
        query: RagQuery,
        hits: list[SearchHit],
        *,
        query_started_at: float | None = None,
    ) -> ChatResult:
        scaffold = self._started_scaffold()
        separator = self._started_separator()
        request_started = time.perf_counter()
        ordered_chunk_ids = reverse_ranked_ids(hits)
        layout = blend_memory_layout(
            scaffold, [self._chunk_token_ids(chunk_id) for chunk_id in ordered_chunk_ids], separator
        )
        query_tokens = self._query_tokens(layout.token_ids, query)
        self._require_budget(layout.token_ids, query_tokens)
        prompt = build_prompt_tokens(layout.token_ids, query_tokens)
        reset_load_stats()
        output = self._generate(prompt)
        stats = snapshot_load_stats()
        matched = int(stats["tokens_matched"])
        if matched < layout.expected_matched_tokens:
            logger.warning(
                "LMCache matched {} of the {} expected tokens for query_id={}",
                matched,
                layout.expected_matched_tokens,
                query.query_id,
            )
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(prompt.memory_token_ids),
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_selected_chunk_ids": ordered_chunk_ids,
            "kv_block_size": self.config.kv_block_size,
            "kv_prefix_caching": 0,
            "retrieved_chunk_count": len(hits),
            "lmcache_separator_tokens": len(separator),
            "lmcache_segment_count": layout.segment_count,
            "lmcache_expected_matched_tokens": layout.expected_matched_tokens,
            "lmcache_matched_tokens": matched,
            "lmcache_load_time_ms": float(stats["load_time_ms"]),
            **generation_latency_metrics(
                output,
                request_started=request_started,
                query_started_at=query_started_at,
                engine_ttft_key="lmcache_engine_time_to_first_token_ms",
            ),
        }
        return ChatResult(content=output.text, ttft_ms=output.ttft_ms, metrics=metrics)

    def close(self) -> None:
        shutdown()
        super().close()
        self._separator = None
        empty_cuda_cache()

    def _engine_config(self) -> EngineConfig:
        return engine_config_for_lmcache(self.config.engine_config())

    def _kv_transfer_config(self) -> Mapping[str, Any]:
        return lmcache_kv_transfer_config(self.config.lmcache_settings())

    def _after_start(self) -> None:
        """Register the model, check the separator, then store every chunk's KV once."""
        settings = self.config.lmcache_settings()
        register_model(self._engine.llm)
        separator = require_separator_parity(self._engine.tokenizer, settings.blend_special_str)
        scaffold = self._started_scaffold()
        prompts = warm_up_prompts(
            scaffold.header_token_ids, (chunk.token_ids for chunk in self._chunks), separator
        )
        bytes_per_token = kv_bytes_per_token()
        required = require_cache_capacity(
            warm_tokens=prompt_token_count(prompts),
            tail_tokens=TAIL_ALLOWANCE_TOKENS * self._query_count,
            bytes_per_token=bytes_per_token,
            max_local_cpu_size_gb=settings.max_local_cpu_size_gb,
            what="The RAG corpus",
        )
        logger.info(
            "Warming LMCache with {} prompts ({} tokens) for {} corpus chunks",
            len(prompts),
            prompt_token_count(prompts),
            len(self._chunks),
        )
        stats = warm_prompts(self._engine.llm, prompts, batch_size=WARM_BATCH_SIZE)
        self._separator = separator
        self._warm = {
            **warmup_metrics(stats),
            "lmcache_kv_bytes_per_token": bytes_per_token,
            "lmcache_required_cpu_cache_gb": required / GB,
            "lmcache_max_local_cpu_size_gb": settings.max_local_cpu_size_gb,
            "lmcache_separator_tokens": len(separator),
        }

    def _started_separator(self) -> list[int]:
        if self._separator is None:
            raise RuntimeError(
                f"{type(self).__name__}.start_llm() must be called before answering."
            )
        return self._separator
