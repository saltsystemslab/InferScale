from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from typing import Any

from ..clients import ChatResult
from ..config import BenchmarkConfig
from ..data import ConversationSample, QuestionAnswer
from ..vector_types import SearchHit
from .chunk_cache import (
    cache_meta,
    cache_path_for,
    load_sample_chunks,
    save_sample_chunks,
    scaffold_chunks_match,
)
from .chunk_store import build_chunk_store
from .chunked_rope import ChunkedRopeEncoder, ChunkedRopeSampleComposer, load_encoder_tokenizer
from .context import memory_context_metrics, unique_memory_facts
from .gpu_registry import (
    clear_namespace,
    drop_namespace,
    get_gpu_memory_store,
    namespace_stats,
    register_user_memory,
    remove_user_memory,
)
from .prompting import (
    build_kv_equivalence_prompt_from_query_tokens,
    build_kv_query_tokens_for_memory,
    build_memory_prompt_token_ids,
    calculate_memory_token_budget,
    extract_memory_scaffold_token_ids,
)
from .sample_cache import GpuSampleCacheStore
from .vllm_metrics import require_engine_ttft_ms
from .vllm_runtime import (
    build_strict_gpu_kv_transfer_config,
    common_vllm_kwargs,
    empty_cuda_cache,
    force_vllm_inprocess_mode,
)

logger = logging.getLogger(__name__)


class VLLMChunkedKVAnswerClient:
    """In-process vLLM answer client using strict GPU chunked-RoPE KV injection."""

    def __init__(self, config: BenchmarkConfig) -> None:
        force_vllm_inprocess_mode()
        self.config = config
        self.namespace = f"{config.run_id}-{uuid.uuid4().hex}"
        self.active_user_id = f"{self.namespace}-active"
        # The namespace registry holds the in-flight composed memory for the
        # connector handoff and is always GPU-resident; it is NOT the memory
        # store. --kv-store-backend selects where the fact-chunk corpus
        # lives (the per-sample chunk store).
        get_gpu_memory_store(self.namespace, backend="gpu")
        self._llm: Any | None = None
        self._tokenizer: Any | None = None
        self._sampling_cls: Any | None = None
        self._sample_caches = GpuSampleCacheStore()
        self._encoder: ChunkedRopeEncoder | None = None
        self._standalone_tokenizer: Any | None = None

    def precompute_sample_cache(
        self,
        sample: ConversationSample,
        facts: Sequence[SearchHit],
    ) -> dict[str, Any]:
        force_vllm_inprocess_mode()
        if self._llm is not None:
            raise RuntimeError(
                "GPU-resident KV precompute must run before vLLM is started. "
                "Precompute all needed samples before starting the single vLLM instance."
            )
        sample_key = id(sample)
        if self._sample_caches.active_sample_key == sample_key:
            self.close_sample()
        else:
            self._sample_caches.release(sample_key)
        logger.info(
            "Precomputing GPU-resident KV cache sample_id=%s facts=%d context_window=%d",
            sample.sample_id,
            len(facts),
            self.config.context_window,
        )
        started = time.perf_counter()
        composer: ChunkedRopeSampleComposer | None = None
        cache_hit = False
        cache_path = None
        payload_meta = None
        kv_facts = unique_memory_facts(facts)
        # The fact-chunk corpus lives in the backend-selected store; under
        # the cpu backend the disk cache loads straight to host and gets
        # pinned by the store, never touching the GPU.
        chunk_device = "cpu" if self.config.kv_store_backend == "cpu" else self.config.kv_device
        chunk_store = build_chunk_store(
            self.config.kv_store_backend,
            device=self.config.kv_device,
            top_k=self.config.top_k,
            staging_slots=self.config.kv_staging_slots,
        )
        if self.config.kv_chunk_cache_enabled:
            key_kwargs = dict(
                model=self.config.model,
                dtype=self.config.kv_dtype,
                context_window=self.config.context_window,
                max_position=self.config.kv_max_position,
                block_size=self.config.kv_block_size,
                sample=sample,
                facts=kv_facts,
            )
            cache_path = cache_path_for(**key_kwargs)
            payload_meta = cache_meta(**key_kwargs)
            cached = load_sample_chunks(
                cache_path,
                device=chunk_device,
                scaffold_device=self.config.kv_device,
                expected_meta=payload_meta,
                expected_fact_ids=[fact.memory_id for fact in kv_facts],
            )
            if cached is not None:
                tokenizer = self._encoder_tokenizer()
                scaffold = extract_memory_scaffold_token_ids(
                    tokenizer,
                    sample,
                    block_size=self.config.kv_block_size,
                )
                if scaffold_chunks_match(cached.scaffold_chunks, scaffold):
                    composer = ChunkedRopeSampleComposer.from_cached(
                        encoder=ChunkedRopeEncoder.from_tables(
                            model=self.config.model,
                            device=self.config.kv_device,
                            max_position=self.config.kv_max_position,
                            tokenizer=tokenizer,
                            cos_table=cached.cos_table,
                            sin_table=cached.sin_table,
                        ),
                        cached=cached,
                        sample=sample,
                        facts=facts,
                        context_window=self.config.context_window,
                        block_size=self.config.kv_block_size,
                        chunk_store=chunk_store,
                    )
                    cache_hit = True
                    logger.info(
                        "KV chunk cache hit sample_id=%s (%s)", sample.sample_id, cache_path.name
                    )
                else:
                    logger.warning(
                        "KV chunk cache scaffold mismatch for sample_id=%s; re-encoding",
                        sample.sample_id,
                    )
        if composer is None:
            try:
                encoder = self._ensure_encoder()
                composer = ChunkedRopeSampleComposer(
                    encoder=encoder,
                    context_window=self.config.context_window,
                    block_size=self.config.kv_block_size,
                    chunk_store=chunk_store,
                )
                composer.encode_sample(sample, facts)
            except RuntimeError as exc:
                if composer is not None:
                    composer.close()
                raise RuntimeError(
                    f"GPU-resident KV precompute failed for sample_id={sample.sample_id}. "
                    "No CPU or disk KV fallback was used."
                ) from exc
            except BaseException:
                if composer is not None:
                    composer.close()
                raise
            if self.config.kv_chunk_cache_enabled and cache_path is not None:
                try:
                    save_sample_chunks(
                        cache_path,
                        meta=payload_meta,
                        fact_chunks=composer.chunks,
                        scaffold_chunks={
                            "header": composer.header_chunk,
                            "memory_list_header": composer.memory_list_header_chunk,
                            "empty_memory": composer.empty_memory_chunk,
                            "footer": composer.footer_chunk,
                        },
                        cos_table=encoder.cos_table,
                        sin_table=encoder.sin_table,
                    )
                except Exception as exc:
                    # The cache is an accelerator, never a correctness
                    # dependency; a full disk must not kill the run.
                    logger.warning("Failed to save KV chunk cache %s: %s", cache_path, exc)
            # The disk save above needed the real tensors; the store owns
            # them from here (D2H into pinned buffers under the cpu backend).
            composer.move_chunks_to_store()

        sample_metrics = composer.cache_stats()
        sample_metrics.update(
            {
                "kv_precompute_time_ms": (time.perf_counter() - started) * 1000,
                "kv_chunk_cache_residency_is_gpu": int(self.config.kv_store_backend == "gpu"),
                "kv_chunk_cache_hit": int(cache_hit),
            }
        )
        self._sample_caches.put(sample_key, composer, sample_metrics)
        logger.info(
            "Precomputed GPU-resident KV cache sample_id=%s chunks=%s tokens=%s layers=%s gpu_mb=%.1f devices=%s",
            sample.sample_id,
            sample_metrics.get("kv_precomputed_chunks", 0),
            sample_metrics.get("kv_precomputed_tokens", 0),
            sample_metrics.get("kv_precomputed_layers", 0),
            sample_metrics.get("kv_precomputed_gpu_mb", 0.0),
            sample_metrics.get("kv_precomputed_devices", ""),
        )
        return dict(sample_metrics)

    def start_llm(self) -> None:
        force_vllm_inprocess_mode()
        if self._llm is not None:
            return
        if not self._sample_caches:
            raise RuntimeError("At least one GPU-resident KV sample cache must be precomputed before starting vLLM.")
        self._release_encoder_model()

        from vllm import LLM, SamplingParams

        try:
            self._sampling_cls = SamplingParams
            self._llm = LLM(
                **common_vllm_kwargs(self.config),
                kv_transfer_config=build_strict_gpu_kv_transfer_config(
                    connector_module=self.config.kv_connector_module,
                    namespace=self.namespace,
                    default_user_id=self.active_user_id,
                    # The connector's namespace registry holds the in-flight
                    # composed memory and is always GPU-resident; the corpus
                    # backend is the chunk store's concern.
                    store_backend="gpu",
                ),
            )
            self._tokenizer = self._llm.get_tokenizer()
        except Exception:
            self.close()
            raise

    def prepare_sample(self, sample: ConversationSample) -> None:
        sample_key = id(sample)
        if self._sample_caches.active_sample_key is not None and self._sample_caches.active_sample_key != sample_key:
            self.close_sample()
        self._sample_caches.prepare(sample_key, sample.sample_id)
        logger.info("Preparing GPU-resident KV sample_id=%s", sample.sample_id)

    def answer_with_retrieved_memory(
        self,
        *,
        sample: ConversationSample,
        qa: QuestionAnswer,
        hits: list[SearchHit],
        max_tokens: int,
        temperature: float,
        top_p: float,
        ttft_started_at: float | None = None,
        query_started_at: float | None = None,
    ) -> ChatResult:
        if self._llm is None or self._tokenizer is None or self._sampling_cls is None:
            raise RuntimeError("VLLMChunkedKVAnswerClient.prepare_sample() must be called before answering.")
        composer = self._sample_caches.active_composer
        if composer is None:
            raise RuntimeError(f"Strict GPU KV cache sample_id={sample.sample_id} was not prepared.")
        if self._sample_caches.active_sample_key != id(sample):
            raise RuntimeError(f"Strict GPU KV sample_id={sample.sample_id} is not the active prepared sample.")

        request_started = ttft_started_at if ttft_started_at is not None else time.perf_counter()
        scaffold = extract_memory_scaffold_token_ids(
            self._tokenizer,
            sample,
            block_size=self.config.kv_block_size,
        )
        query_tokens = build_kv_query_tokens_for_memory(
            self._tokenizer,
            scaffold.header_token_ids,
            sample,
            qa,
            memory_scaffold=scaffold,
        )
        memory_token_budget = calculate_memory_token_budget(
            self._tokenizer,
            sample,
            qa,
            memory_prefix_token_ids=scaffold.header_token_ids,
            max_position=self.config.kv_max_position,
            max_model_len=self.config.kv_max_model_len,
            max_answer_tokens=max_tokens,
            query_tokens=query_tokens,
            memory_scaffold=scaffold,
        )
        # Chunk staging during composition is the only H2D traffic under the
        # cpu backend (the composed memory is GPU-resident and injects
        # GPU->paged-cache), so this question's transfer metrics are the
        # before/after deltas on the chunk store's counters.
        chunk_store = getattr(composer, "chunk_store", None)
        chunk_totals_before = (
            chunk_store.transfer_totals() if chunk_store is not None else {}
        )
        composed = composer.compose(hits, memory_token_budget=memory_token_budget)
        chunk_totals_after = (
            chunk_store.transfer_totals() if chunk_store is not None else {}
        )
        # Token-equivalence verification is benchmark bookkeeping, not part of the
        # serving path, so its cost is timed separately and excluded from latency.
        verify_started = time.perf_counter()
        live_memory = build_memory_prompt_token_ids(
            self._tokenizer,
            sample,
            hits,
            context_window=self.config.context_window,
            memory_token_budget=memory_token_budget,
            memory_scaffold=scaffold,
        )
        _require_same_memory_token_ids(composed.token_ids, live_memory.token_ids)
        verify_ms = (time.perf_counter() - verify_started) * 1000
        user_id = self.active_user_id
        # Registration hands the composed memory to the GPU-resident
        # in-flight registry (a dict insert, no transfer); timed separately
        # and excluded from the latency metrics like verify.
        store_write_started = time.perf_counter()
        register_user_memory(
            self.namespace,
            user_id=user_id,
            kv_by_layer=composed.kv_by_layer,
            num_tokens=composed.num_tokens,
            token_ids=composed.token_ids,
        )
        store_write_ms = (time.perf_counter() - store_write_started) * 1000
        excluded_prep_ms = verify_ms + store_write_ms
        try:
            prompt = build_kv_equivalence_prompt_from_query_tokens(
                list(composed.token_ids),
                query_tokens,
            )
            metrics: dict[str, Any] = {
                **self._sample_caches.active_metrics,
                "kv_memory_tokens": composed.num_tokens,
                "kv_compose_time_ms": composed.compose_time_ms,
                "kv_verify_time_ms": verify_ms,
                "kv_store_write_time_ms": store_write_ms,
                "kv_context_window": composed.context_window,
                "kv_query_tokens": len(prompt.query_token_ids),
                "kv_query_bos_stripped": int(prompt.stripped_query_bos),
                "kv_selected_fact_ids": composed.selected_fact_ids,
                "kv_block_size": self.config.kv_block_size,
                "kv_prefix_caching": int(self.config.kv_enable_prefix_caching),
                "kv_loaded_memory_tokens": composed.loaded_memory_tokens,
                "kv_recomputed_memory_tail_tokens": composed.recomputed_memory_tail_tokens,
                "kv_fact_tokens_end": composed.fact_tokens_end,
                **memory_context_metrics(composed.fact_plan),
            }
            sampling = self._sampling_cls(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            generate_started = time.perf_counter()
            outputs = self._llm.generate(
                [{"prompt_token_ids": prompt.prompt_token_ids}],
                sampling,
                use_tqdm=False,
            )
            finished = time.perf_counter()
            generate_ms = (finished - generate_started) * 1000
            total_ms = max(0.0, (finished - request_started) * 1000 - excluded_prep_ms)
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
            text = outputs[0].outputs[0].text.strip()
            stats = namespace_stats(self.namespace)
            chunk_stats = chunk_store.get_stats() if chunk_store is not None else {}
            chunk_h2d_ms = float(chunk_totals_after.get("total_h2d_latency_ms", 0.0)) - float(
                chunk_totals_before.get("total_h2d_latency_ms", 0.0)
            )
            chunk_stall_ms = float(chunk_totals_after.get("total_staging_stall_ms", 0.0)) - float(
                chunk_totals_before.get("total_staging_stall_ms", 0.0)
            )
            chunk_bytes = int(chunk_totals_after.get("total_bytes_transferred", 0)) - int(
                chunk_totals_before.get("total_bytes_transferred", 0)
            )
            metrics.update(
                {
                    "answer_generate_time_ms": generate_ms,
                    "answer_total_time_ms": total_ms,
                    "kv_store_gpu_mb": stats.get("total_gpu_mb", 0.0),
                    "kv_store_host_mb": chunk_stats.get("total_host_mb", 0.0),
                    "kv_h2d_latency_ms": chunk_h2d_ms,
                    "kv_h2d_bytes": chunk_bytes,
                    "kv_h2d_overlap_ratio": (
                        max(0.0, min(1.0, 1.0 - chunk_stall_ms / chunk_h2d_ms))
                        if chunk_h2d_ms > 0
                        else 0.0
                    ),
                    "kv_staging_stall_ms": chunk_stall_ms,
                }
            )
            return ChatResult(
                content=text,
                ttft_ms=ttft_ms,
                metrics=metrics,
            )
        finally:
            remove_user_memory(self.namespace, user_id)

    def close_sample(self) -> None:
        self._sample_caches.release_active()
        clear_namespace(self.namespace)
        empty_cuda_cache()

    def close(self) -> None:
        self._sample_caches.release_all()
        clear_namespace(self.namespace)
        self._close_encoder()
        if self._llm is not None:
            del self._llm
            self._llm = None
        self._tokenizer = None
        self._sampling_cls = None
        drop_namespace(self.namespace)
        empty_cuda_cache(collect_ipc=True)

    def _ensure_encoder(self) -> ChunkedRopeEncoder:
        if self._encoder is None:
            self._encoder = ChunkedRopeEncoder(
                model=self.config.model,
                dtype=self.config.kv_dtype,
                device=self.config.kv_device,
                max_position=self.config.kv_max_position,
            )
        return self._encoder

    def _encoder_tokenizer(self) -> Any:
        """Encoder-side tokenizer without forcing an HF weight load.

        Reuses the live encoder's tokenizer when one exists; otherwise loads
        the tokenizer standalone (memoized) so fully cached runs never touch
        the model weights.
        """
        if self._encoder is not None and self._encoder.tokenizer is not None:
            return self._encoder.tokenizer
        if self._standalone_tokenizer is None:
            self._standalone_tokenizer = load_encoder_tokenizer(self.config.model)
        return self._standalone_tokenizer

    def _release_encoder_model(self) -> None:
        if self._encoder is not None:
            self._encoder.release_model()

    def _close_encoder(self) -> None:
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None


def _first_token_mismatch(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _require_same_memory_token_ids(
    precomputed_token_ids: list[int],
    live_token_ids: list[int],
) -> None:
    if precomputed_token_ids == live_token_ids:
        return
    mismatch_index = _first_token_mismatch(
        precomputed_token_ids,
        live_token_ids,
    )
    raise RuntimeError(
        "Precomputed Hugging Face memory tokens differ from the live vLLM "
        f"tokenizer at index={mismatch_index}: "
        f"precomputed_length={len(precomputed_token_ids)} "
        f"live_length={len(live_token_ids)}."
    )
