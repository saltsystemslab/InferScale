"""The InferScale answer client: fetch, compose, and inject the cached chunk KV per query.

The injection path is the memory benchmark's: composition concatenates
pre-RoPE chunks and applies RoPE once at the new virtual positions,
register_user_memory hands the composed memory to the namespace registry,
and the KV connector scatters it into vLLM's paged cache by prompt-prefix
token match. Only chunk residency differs: the corpus KV is resident in host
RAM (cpu_chunk_store.py), loaded once from the precompute cache, because the
full corpus does not fit GPU memory at MultiHop-RAG scale.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from benchmarks.common.answer_timing import generation_latency_metrics
from benchmarks.common.clients import ChatResult
from benchmarks.rag.config import RagBenchConfig
from benchmarks.rag.cpu_chunk_store import CpuChunkStore
from benchmarks.rag.data_types import RagChunk, RagPromptProfile, RagQuery
from benchmarks.rag.engine_client import RagEngineClient
from benchmarks.rag.kv_cache import (
    CachedRagScaffold,
    RagChunkCache,
    load_tables_and_scaffold,
    rag_scaffold_chunks_match,
)
from benchmarks.rag.prompting import extract_rag_scaffold_token_ids
from inferscale.v1.kv.compose import reverse_ranked_ids
from inferscale.v1.kv.encoder import ChunkedRopeEncoder, load_encoder_tokenizer
from inferscale.v1.kv.prompt import (
    ScaffoldTokens,
    block_aligned_prefix,
    build_memory_token_ids,
    build_prompt_tokens,
    require_identical_token_ids,
)
from inferscale.v1.kv.registry import (
    clear_namespace,
    drop_namespace,
    get_gpu_memory_store,
    namespace_stats,
    register_user_memory,
    remove_user_memory,
)
from inferscale.v1.serving.vllm import build_kv_transfer_config, empty_cuda_cache
from inferscale.v1.types import KVChunk, SearchHit


@dataclass(slots=True)
class ComposedPassages:
    """The retrieved chunks between the scaffold header and footer, as KV parts and token ids."""

    parts: list[KVChunk]
    token_ids: list[int]
    fetch_ms: float


class RagKvAnswerClient(RagEngineClient):
    label = "kv-injection"

    def __init__(
        self,
        config: RagBenchConfig,
        *,
        chunks: Sequence[RagChunk],
        cache: RagChunkCache,
        prompt_profile: RagPromptProfile,
    ) -> None:
        super().__init__(
            config,
            chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
            prompt_profile=prompt_profile,
        )
        self._chunks = list(chunks)
        self._cache = cache
        encoder_tokenizer = load_encoder_tokenizer(config.model)
        cached = _cached_scaffold(
            cache,
            config,
            expected=extract_rag_scaffold_token_ids(
                encoder_tokenizer,
                system_prompt=prompt_profile.system_prompt,
                block_size=config.kv_block_size,
            ),
        )
        # Answer runs never load HF encoder weights: an encoder built from the
        # cached RoPE tables only composes.
        self._encoder = ChunkedRopeEncoder.from_tables(
            model=config.model,
            device=config.kv_device,
            max_position=config.kv_max_position,
            tokenizer=encoder_tokenizer,
            cos_table=cached.cos_table,
            sin_table=cached.sin_table,
        )
        self._scaffold_chunks = cached.scaffold_chunks
        self._store: CpuChunkStore | None = None
        self.namespace = f"{config.run_id}-{uuid.uuid4().hex}"
        self.active_user_id = f"{self.namespace}-active"
        # The namespace registry holds the in-flight composed memory for the
        # connector handoff and is always GPU-resident.
        get_gpu_memory_store(self.namespace, backend="gpu")

    def answer(
        self,
        query: RagQuery,
        hits: list[SearchHit],
        *,
        query_started_at: float | None = None,
    ) -> ChatResult:
        scaffold = self._started_scaffold()
        request_started = time.perf_counter()
        ordered_chunk_ids = reverse_ranked_ids(hits)
        passages = self._fetch_passages(ordered_chunk_ids)
        query_tokens = self._query_tokens(passages.token_ids, query)
        self._require_budget(passages.token_ids, query_tokens)
        loaded_tokens, tail_tokens = block_aligned_prefix(
            len(passages.token_ids),
            len(self._scaffold_chunks["footer"].token_ids),
            self.config.kv_block_size,
        )
        kv_by_layer, compose_ms = self._compose(passages.parts)
        verify_ms = self._verify(scaffold, ordered_chunk_ids, passages.token_ids)
        store_write_ms = self._register(kv_by_layer, passages.token_ids)
        try:
            # Read while the composed memory is registered: removal empties the namespace.
            store_gpu_mb = namespace_stats(self.namespace).get("total_gpu_mb", 0.0)
            prompt = build_prompt_tokens(passages.token_ids, query_tokens)
            output = self._generate(prompt)
        finally:
            remove_user_memory(self.namespace, self.active_user_id)
        metrics: dict[str, Any] = {
            "kv_memory_tokens": len(passages.token_ids),
            "kv_fetch_time_ms": passages.fetch_ms,
            "kv_compose_time_ms": compose_ms,
            "kv_verify_time_ms": verify_ms,
            "kv_store_write_time_ms": store_write_ms,
            "kv_query_tokens": len(prompt.query_token_ids),
            "kv_query_bos_stripped": int(prompt.stripped_query_bos),
            "kv_selected_chunk_ids": ordered_chunk_ids,
            "kv_block_size": self.config.kv_block_size,
            "kv_context_window": self.config.context_window,
            "kv_loaded_memory_tokens": loaded_tokens,
            "kv_recomputed_memory_tail_tokens": tail_tokens,
            "retrieved_chunk_count": len(hits),
            # Parity verification and the registry handoff are benchmark
            # bookkeeping, not part of the serving path, so they are excluded.
            **generation_latency_metrics(
                output,
                request_started=request_started,
                query_started_at=query_started_at,
                engine_ttft_key="kv_engine_time_to_first_token_ms",
                excluded_ms=verify_ms + store_write_ms,
            ),
            "kv_store_gpu_mb": store_gpu_mb,
        }
        return ChatResult(content=output.text, ttft_ms=output.ttft_ms, metrics=metrics)

    def setup_stats(self) -> dict[str, Any]:
        return self._store.stats() if self._store is not None else {}

    def close(self) -> None:
        clear_namespace(self.namespace)
        super().close()
        self._encoder.close()
        if self._store is not None:
            self._store.close()
            self._store = None
        drop_namespace(self.namespace)
        empty_cuda_cache(collect_ipc=True)

    def _prepare(self) -> None:
        # The heavy step: the whole corpus chunk KV into host RAM, once.
        self._store = CpuChunkStore(self._cache, chunks=self._chunks, device=self.config.kv_device)

    def _kv_transfer_config(self) -> Mapping[str, Any]:
        return build_kv_transfer_config(
            connector_module=self.config.kv_connector_module,
            namespace=self.namespace,
            default_user_id=self.active_user_id,
            store_backend="gpu",
        )

    def _fetch_passages(self, ordered_chunk_ids: list[str]) -> ComposedPassages:
        if self._store is None:
            raise RuntimeError(
                f"{type(self).__name__}.start_llm() must be called before answering."
            )
        fetch_started = time.perf_counter()
        encoded_chunks = self._store.fetch(ordered_chunk_ids)
        fetch_ms = (time.perf_counter() - fetch_started) * 1000
        parts = [self._scaffold_chunks["header"]]
        parts.extend(encoded_chunks or [self._scaffold_chunks["empty_passages"]])
        parts.append(self._scaffold_chunks["footer"])
        token_ids = [token for part in parts for token in part.token_ids]
        return ComposedPassages(parts=parts, token_ids=token_ids, fetch_ms=fetch_ms)

    def _compose(self, parts: list[KVChunk]) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        kv_by_layer = self._encoder.compose(parts)
        return kv_by_layer, (time.perf_counter() - started) * 1000

    def _verify(
        self, scaffold: ScaffoldTokens, ordered_chunk_ids: list[str], memory_token_ids: list[int]
    ) -> float:
        """Check the composed token ids against what the live tokenizer would produce."""
        started = time.perf_counter()
        live_token_ids = build_memory_token_ids(
            scaffold, [self._chunk_token_ids(chunk_id) for chunk_id in ordered_chunk_ids]
        )
        require_identical_token_ids(memory_token_ids, live_token_ids)
        return (time.perf_counter() - started) * 1000

    def _register(self, kv_by_layer: dict[str, Any], memory_token_ids: list[int]) -> float:
        started = time.perf_counter()
        register_user_memory(
            self.namespace,
            user_id=self.active_user_id,
            kv_by_layer=kv_by_layer,
            num_tokens=len(memory_token_ids),
            token_ids=memory_token_ids,
        )
        return (time.perf_counter() - started) * 1000


def _cached_scaffold(
    cache: RagChunkCache, config: RagBenchConfig, *, expected: ScaffoldTokens
) -> CachedRagScaffold:
    """The precomputed RoPE tables and scaffold chunks, which must match the live scaffold."""
    tables_path = cache.tables_path(config.kv_block_size)
    cached = load_tables_and_scaffold(
        tables_path,
        expected_meta=cache.key.tables_meta(block_size=config.kv_block_size),
        device=config.kv_device,
    )
    if cached is None:
        raise RuntimeError(
            f"RAG KV tables/scaffold cache is missing or stale at {tables_path}. "
            "Run the precompute-kv stage with the same model, chunk size, and "
            "context window first."
        )
    if not rag_scaffold_chunks_match(cached.scaffold_chunks, expected):
        raise RuntimeError(
            f"RAG KV scaffold cache at {tables_path} does not match the live tokenizer's "
            "scaffold token ids; re-run the precompute-kv stage."
        )
    return cached
