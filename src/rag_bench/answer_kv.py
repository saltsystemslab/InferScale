from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from locomo_jasper_bench.clients import ChatResult
from locomo_jasper_bench.kv.chunked_rope import ChunkedRopeEncoder, load_encoder_tokenizer
from locomo_jasper_bench.kv.gpu_registry import (
    clear_namespace,
    drop_namespace,
    get_gpu_memory_store,
    namespace_stats,
    register_user_memory,
    remove_user_memory,
)
from locomo_jasper_bench.kv.prompting import build_kv_equivalence_prompt_from_query_tokens
from locomo_jasper_bench.kv.vllm_metrics import require_engine_ttft_ms
from locomo_jasper_bench.kv.vllm_runtime import (
    build_strict_gpu_kv_transfer_config,
    common_vllm_kwargs,
    empty_cuda_cache,
    force_vllm_inprocess_mode,
)
from locomo_jasper_bench.vector_types import SearchHit

from .config import RagBenchConfig
from .data_types import RagChunk, RagPromptProfile, RagQuery
from .kv_cache import (
    CpuChunkStore,
    load_tables_and_scaffold,
    rag_scaffold_chunks_match,
    tables_meta,
    tables_scaffold_path,
)
from .prompting import (
    build_rag_memory_token_ids,
    build_rag_query_tokens,
    calculate_rag_memory_budget,
    extract_rag_scaffold_token_ids,
    require_identical_token_ids,
    require_memory_within_budget,
    reverse_ranked_chunk_ids,
)


class RagKvAnswerClient:
    """InferScale answer client: per-query fetch, compose, and inject cached chunk KV.

    The injection path is identical to the LoCoMo pipeline: compose_chunks
    concatenates pre-RoPE chunks and applies RoPE once at the new virtual
    positions, register_user_memory hands the composed memory to the namespace
    registry, and the MemoryKVConnector scatters it into vLLM's paged cache by
    prompt-prefix token match. Only chunk residency differs: the corpus chunk
    KV is fully resident in host RAM (CpuChunkStore, the cpu store backend,
    loaded once from the precompute cache) instead of a per-sample GPU store,
    because the full corpus KV does not fit GPU HBM at MultiHop-RAG scale.
    """

    def __init__(
        self,
        config: RagBenchConfig,
        *,
        chunks: Sequence[RagChunk],
        cache_dir: Path,
        meta_base: Mapping[str, Any],
        prompt_profile: RagPromptProfile,
    ) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self._chunks_by_id: Mapping[str, RagChunk] = {
            chunk.chunk_id: chunk for chunk in chunks
        }
        self._prompt_profile = prompt_profile
        self.namespace = f"{config.run_id}-{uuid.uuid4().hex}"
        self.active_user_id = f"{self.namespace}-active"
        # The namespace registry holds the in-flight composed memory for the
        # connector handoff and is always GPU-resident.
        get_gpu_memory_store(self.namespace, backend="gpu")

        encoder_tokenizer = load_encoder_tokenizer(config.model)
        expected_scaffold = extract_rag_scaffold_token_ids(
            encoder_tokenizer,
            system_prompt=prompt_profile.system_prompt,
            block_size=config.kv_block_size,
        )
        tables_path = tables_scaffold_path(cache_dir, config.kv_block_size)
        cached = load_tables_and_scaffold(
            tables_path,
            expected_meta=tables_meta(meta_base, block_size=config.kv_block_size),
            device=config.kv_device,
        )
        if cached is None:
            raise RuntimeError(
                f"RAG KV tables/scaffold cache is missing or stale at {tables_path}. "
                "Run rag-jasper-bench --precompute-kv-only with the same --model, "
                "--chunk-size, and --context-window first."
            )
        if not rag_scaffold_chunks_match(cached.scaffold_chunks, expected_scaffold):
            raise RuntimeError(
                f"RAG KV scaffold cache at {tables_path} does not match the live tokenizer's "
                "scaffold token ids; re-run rag-jasper-bench --precompute-kv-only."
            )
        # Cache-complete runs never load HF encoder weights: the from_tables
        # encoder only composes.
        self._encoder = ChunkedRopeEncoder.from_tables(
            model=config.model,
            device=config.kv_device,
            max_position=config.kv_max_position,
            tokenizer=encoder_tokenizer,
            cos_table=cached.cos_table,
            sin_table=cached.sin_table,
        )
        self._header_chunk = cached.scaffold_chunks["header"]
        self._empty_chunk = cached.scaffold_chunks["empty_passages"]
        self._footer_chunk = cached.scaffold_chunks["footer"]
        # The heavy step: load the full corpus chunk KV into host RAM once.
        self._store = CpuChunkStore(cache_dir, meta_base=meta_base, chunks=chunks)
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_cls: Any | None = None
        self._live_scaffold: Any | None = None

    def start_llm(self) -> None:
        if self._llm is not None:
            return
        force_vllm_inprocess_mode()
        from vllm import LLM, SamplingParams

        try:
            self._sampling_cls = SamplingParams
            self._llm = LLM(
                **common_vllm_kwargs(self.config),
                kv_transfer_config=build_strict_gpu_kv_transfer_config(
                    connector_module=self.config.kv_connector_module,
                    namespace=self.namespace,
                    default_user_id=self.active_user_id,
                    store_backend="gpu",
                ),
            )
            self._tokenizer = self._llm.get_tokenizer()
            self._live_scaffold = extract_rag_scaffold_token_ids(
                self._tokenizer,
                system_prompt=self._prompt_profile.system_prompt,
                block_size=self.config.kv_block_size,
            )
        except Exception:
            self.close()
            raise
        logger.info(
            "Started vLLM KV answer engine model={} namespace={}",
            self.config.model,
            self.namespace,
        )

    def answer(
        self,
        query: RagQuery,
        hits: list[SearchHit],
        *,
        query_started_at: float | None = None,
    ) -> ChatResult:
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
            raise RuntimeError("RagKvAnswerClient.start_llm() must be called before answering.")
        request_started = time.perf_counter()

        ordered_chunk_ids = reverse_ranked_chunk_ids(hits)
        fetch_started = time.perf_counter()
        encoded_chunks = self._store.fetch(ordered_chunk_ids)
        fetch_ms = (time.perf_counter() - fetch_started) * 1000

        parts = [self._header_chunk]
        parts.extend(encoded_chunks if encoded_chunks else [self._empty_chunk])
        parts.append(self._footer_chunk)
        memory_token_ids: list[int] = []
        for part in parts:
            memory_token_ids.extend(part.token_ids)

        query_tokens = build_rag_query_tokens(
            self._tokenizer,
            memory_token_ids,
            query,
            answer_instruction=self._prompt_profile.answer_instruction,
        )
        memory_token_budget = calculate_rag_memory_budget(
            query_token_count=len(query_tokens.token_ids),
            max_position=self.config.kv_max_position,
            max_model_len=self.config.kv_max_model_len,
            max_answer_tokens=self.config.max_answer_tokens,
        )
        require_memory_within_budget(
            len(memory_token_ids),
            memory_token_budget,
            top_k=self.config.top_k,
            chunk_size=self.config.chunk_size,
        )
        # The block-aligned KV prefix must cover every injected chunk token;
        # only footer padding may fall into the recomputed tail.
        chunk_tokens_end = len(memory_token_ids) - len(self._footer_chunk.token_ids)
        loaded_memory_tokens = (
            len(memory_token_ids) // self.config.kv_block_size
        ) * self.config.kv_block_size
        if loaded_memory_tokens < chunk_tokens_end:
            raise RuntimeError(
                "The block-aligned KV prefix would leave retrieved chunk tokens in the "
                f"recomputed tail: loaded={loaded_memory_tokens} chunks_end={chunk_tokens_end} "
                f"memory={len(memory_token_ids)} block_size={self.config.kv_block_size}."
            )

        compose_started = time.perf_counter()
        kv_by_layer = self._encoder.compose_chunks(parts)
        compose_ms = (time.perf_counter() - compose_started) * 1000

        # Parity verification is benchmark bookkeeping, not part of the serving
        # path; its cost is timed separately and excluded from latency.
        verify_started = time.perf_counter()
        live_memory_token_ids = build_rag_memory_token_ids(
            self._live_scaffold,
            [self._chunk_token_ids(chunk_id) for chunk_id in ordered_chunk_ids],
        )
        require_identical_token_ids(memory_token_ids, live_memory_token_ids)
        verify_ms = (time.perf_counter() - verify_started) * 1000

        store_write_started = time.perf_counter()
        register_user_memory(
            self.namespace,
            user_id=self.active_user_id,
            kv_by_layer=kv_by_layer,
            num_tokens=len(memory_token_ids),
            token_ids=memory_token_ids,
        )
        store_write_ms = (time.perf_counter() - store_write_started) * 1000
        excluded_prep_ms = verify_ms + store_write_ms
        try:
            prompt = build_kv_equivalence_prompt_from_query_tokens(
                memory_token_ids,
                query_tokens,
            )
            metrics: dict[str, Any] = {
                "kv_memory_tokens": len(memory_token_ids),
                "kv_fetch_time_ms": fetch_ms,
                "kv_compose_time_ms": compose_ms,
                "kv_verify_time_ms": verify_ms,
                "kv_store_write_time_ms": store_write_ms,
                "kv_query_tokens": len(prompt.query_token_ids),
                "kv_query_bos_stripped": int(prompt.stripped_query_bos),
                "kv_selected_chunk_ids": ordered_chunk_ids,
                "kv_block_size": self.config.kv_block_size,
                "kv_context_window": self.config.context_window,
                "kv_loaded_memory_tokens": loaded_memory_tokens,
                "kv_recomputed_memory_tail_tokens": len(memory_token_ids) - loaded_memory_tokens,
                "retrieved_chunk_count": len(hits),
            }
            sampling = self._sampling_cls(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_answer_tokens,
            )
            generate_started = time.perf_counter()
            outputs = self._llm.generate(
                [{"prompt_token_ids": prompt.prompt_token_ids}],
                sampling,
                use_tqdm=False,
            )
            finished = time.perf_counter()
            metrics["answer_generate_time_ms"] = (finished - generate_started) * 1000
            metrics["answer_total_time_ms"] = max(
                0.0, (finished - request_started) * 1000 - excluded_prep_ms
            )
            if query_started_at is not None:
                metrics["query_to_answer_ms"] = max(
                    0.0, (finished - query_started_at) * 1000 - excluded_prep_ms
                )
            ttft_ms = require_engine_ttft_ms(outputs[0])
            prep_ms = max(0.0, (generate_started - request_started) * 1000 - excluded_prep_ms)
            metrics["kv_engine_time_to_first_token_ms"] = ttft_ms
            metrics["answer_time_to_first_token_ms"] = prep_ms + ttft_ms
            if query_started_at is not None:
                metrics["query_to_first_token_ms"] = (
                    max(0.0, (generate_started - query_started_at) * 1000 - excluded_prep_ms)
                    + ttft_ms
                )
            metrics["kv_store_gpu_mb"] = namespace_stats(self.namespace).get("total_gpu_mb", 0.0)
            return ChatResult(
                content=outputs[0].outputs[0].text.strip(),
                ttft_ms=ttft_ms,
                metrics=metrics,
            )
        finally:
            remove_user_memory(self.namespace, self.active_user_id)

    def store_stats(self) -> dict[str, Any]:
        return self._store.stats()

    def close(self) -> None:
        clear_namespace(self.namespace)
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._tokenizer = None
        self._sampling_cls = None
        self._live_scaffold = None
        encoder = getattr(self, "_encoder", None)
        if encoder is not None:
            encoder.close()
            self._encoder = None
        drop_namespace(self.namespace)
        empty_cuda_cache(collect_ipc=True)

    def _chunk_token_ids(self, chunk_id: str) -> list[int]:
        chunk = self._chunks_by_id.get(chunk_id)
        if chunk is None:
            raise RuntimeError(f"Retrieved chunk id {chunk_id} is not part of the corpus.")
        return chunk.token_ids
