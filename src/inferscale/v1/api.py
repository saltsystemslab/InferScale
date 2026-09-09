"""The InferScale facade: precompute context, start serving, answer queries.

Phases are strict. Every ``precompute`` call happens before ``start``; after
``start`` only ``retrieve``, ``answer``, and ``query`` are available.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import replace
from itertools import islice
from typing import Any

from .config import InferScaleConfig
from .embedding.openai import OpenAIEmbedder
from .index.jasper import JasperIndex
from .kv.chunk_store import build_chunk_store, chunk_nbytes
from .kv.compose import memory_parts, reverse_ranked_ids
from .kv.corpus import KVCorpus
from .kv.encoder import ChunkedRopeEncoder, encode_scaffold
from .kv.plan import build_encoding_plan
from .kv.prompt import (
    block_aligned_prefix,
    build_prompt_tokens,
    build_query_tokens,
    build_scaffold_tokens,
    memory_token_budget,
    require_memory_within_budget,
)
from .kv.registry import (
    clear_namespace,
    drop_namespace,
    get_gpu_memory_store,
    namespace_stats,
    register_user_memory,
    remove_user_memory,
)
from .kv.tokenization import encode_text_no_special
from .protocols import ChunkStore, Embedder, VectorIndex
from .retrieval import Retriever
from .serving.vllm import (
    VLLMEngine,
    build_kv_transfer_config,
    empty_cuda_cache,
    force_vllm_inprocess_mode,
)
from .types import (
    Chunk,
    ChunkInfo,
    EncodingPlan,
    PrecomputeStats,
    QueryResult,
    RetrievalMetrics,
    SearchHit,
)

_EMPTY_CACHE_EVERY = 25


class InferScale:
    def __init__(
        self,
        config: InferScaleConfig,
        *,
        embedder: Embedder | None = None,
        index: VectorIndex | None = None,
        chunk_store: ChunkStore | None = None,
        engine: VLLMEngine | None = None,
    ) -> None:
        force_vllm_inprocess_mode()
        self._config = config
        self._namespace = f"inferscale-{uuid.uuid4().hex}"
        self._active_user_id = f"{self._namespace}-active"
        # The namespace registry holds the in-flight composed memory for the
        # connector handoff and is always GPU-resident; the chunk store is
        # where the corpus lives.
        get_gpu_memory_store(self._namespace, backend="gpu")
        store = (
            chunk_store
            if chunk_store is not None
            else build_chunk_store(
                config.kv.store_backend,
                device=config.kv.device,
                top_k=config.top_k,
                staging_slots=config.kv.staging_slots,
            )
        )
        self._corpus = KVCorpus(store)
        self._embedder: Embedder = (
            embedder
            if embedder is not None
            else OpenAIEmbedder(model=config.embedding.model, base_url=config.embedding.base_url)
        )
        self._index: VectorIndex = index if index is not None else JasperIndex(config.index)
        self._retriever = Retriever(self._embedder, self._index, batch_size=config.embedding.batch_size)
        self._engine = engine if engine is not None else VLLMEngine(config.model, config.kv.dtype, config.engine)
        self._encoder: ChunkedRopeEncoder | None = None
        self._chunks: dict[str, ChunkInfo] = {}
        self._started = False
        self._closed = False

    @property
    def config(self) -> InferScaleConfig:
        return self._config

    @property
    def tokenizer(self) -> Any:
        """Encoder tokenizer before start(), engine tokenizer after."""
        if self._started:
            return self._engine.tokenizer
        return self._ensure_encoder().tokenizer

    def precompute(self, chunks: Iterable[Chunk]) -> PrecomputeStats:
        """Encode and index chunks in order, continuing the window across calls.

        Each chunk uses up to ``context_window`` preceding chunks as encoding
        context. Only the target chunk's KV is kept for retrieval.
        """
        self._require_open()
        if self._started:
            raise RuntimeError("InferScale is serving; precompute all context before start().")
        chunk_list = list(chunks)
        if not chunk_list:
            return PrecomputeStats(0, 0, 0, 0.0, 0.0, self._corpus.stats())
        self._require_new_ids(chunk_list)

        cfg = self._config
        encoder = self._ensure_encoder()
        # Plan the complete batch before embedding or encoding its chunks.
        plans = self._plans_for(encoder, chunk_list)
        if self._corpus.scaffold is None:
            scaffold_tokens = build_scaffold_tokens(
                encoder.tokenizer,
                system_prompt=cfg.prompt.system_prompt,
                empty_text=cfg.prompt.empty_context_text,
                block_size=cfg.engine.block_size,
            )
            self._corpus.set_scaffold(encode_scaffold(encoder, scaffold_tokens))

        payloads = [
            dict(chunk.payload) if chunk.payload is not None else {"text": chunk.text}
            for chunk in chunk_list
        ]
        embed_ms = self._retriever.add(
            [chunk.id for chunk in chunk_list],
            [chunk.text for chunk in chunk_list],
            payloads,
        )

        encode_started = time.perf_counter()
        token_count = 0
        kv_bytes = 0
        for index, (chunk, plan, payload) in enumerate(zip(chunk_list, plans, payloads), start=1):
            kv_chunk = encoder.encode_plan(plan)
            kv_bytes += chunk_nbytes(kv_chunk)
            token_count += len(kv_chunk.token_ids)
            self._corpus.add(kv_chunk)
            self._chunks[chunk.id] = ChunkInfo(
                id=chunk.id,
                text=chunk.text,
                token_ids=list(plan.target_token_ids),
                context_ids=plan.context_ids,
                context_prefix_tokens=kv_chunk.context_prefix_tokens,
                payload=payload,
            )
            if index % _EMPTY_CACHE_EVERY == 0:
                empty_cuda_cache()
        encode_ms = (time.perf_counter() - encode_started) * 1000
        return PrecomputeStats(
            chunk_count=len(chunk_list),
            token_count=token_count,
            kv_bytes=kv_bytes,
            encode_time_ms=encode_ms,
            embed_time_ms=embed_ms,
            store=self._corpus.stats(),
        )

    def start(self) -> None:
        """Release the encoder weights, finalize the index, and start vLLM."""
        self._require_open()
        if self._started:
            return
        if not self._chunks or self._encoder is None:
            raise RuntimeError("Precompute at least one chunk before start().")
        cfg = self._config
        self._encoder.release_model()
        self._corpus.finalize()
        empty_cuda_cache()
        self._retriever.finalize()
        self._engine.start(
            kv_transfer_config=build_kv_transfer_config(
                connector_module=cfg.engine.connector_module,
                namespace=self._namespace,
                default_user_id=self._active_user_id,
                store_backend="gpu",
            )
        )
        self._started = True

    def retrieve(self, question: str, *, top_k: int | None = None) -> tuple[list[SearchHit], RetrievalMetrics]:
        self._require_serving()
        return self._retriever.search(question, top_k=top_k if top_k is not None else self._config.top_k)

    def answer(
        self,
        question: str | Sequence[dict[str, str]],
        hits: Sequence[SearchHit],
        *,
        query_started_at: float | None = None,
    ) -> QueryResult:
        """Compose the retrieved chunks' KV, inject it, and generate."""
        self._require_serving()
        cfg = self._config
        scaffold = self._corpus.scaffold
        encoder = self._encoder
        if scaffold is None or encoder is None:
            raise RuntimeError("No precomputed context is available.")
        request_started = time.perf_counter()

        ordered_ids = reverse_ranked_ids(hits)
        fetch_started = time.perf_counter()
        selected = self._corpus.fetch(ordered_ids)
        fetch_ms = (time.perf_counter() - fetch_started) * 1000

        parts = memory_parts(scaffold, selected)
        memory_token_ids: list[int] = []
        for part in parts:
            memory_token_ids.extend(part.token_ids)

        messages = (
            [{"role": "user", "content": question}] if isinstance(question, str) else list(question)
        )
        query_tokens = build_query_tokens(self._engine.tokenizer, memory_token_ids, messages)
        budget = memory_token_budget(
            query_token_count=len(query_tokens.token_ids),
            max_position=cfg.kv.max_position,
            max_model_len=cfg.engine.max_model_len,
            max_answer_tokens=cfg.generation.max_tokens,
        )
        require_memory_within_budget(len(memory_token_ids), budget)
        loaded_memory_tokens, recomputed_tail_tokens = block_aligned_prefix(
            len(memory_token_ids),
            len(scaffold.footer.token_ids),
            cfg.engine.block_size,
        )

        compose_started = time.perf_counter()
        try:
            kv_by_layer = encoder.compose(parts)
        finally:
            self._corpus.release(ordered_ids)
        compose_ms = (time.perf_counter() - compose_started) * 1000

        # Registration hands the composed memory to the GPU-resident
        # in-flight registry (a dict insert, no transfer); timed separately
        # and excluded from the latency metrics.
        store_write_started = time.perf_counter()
        register_user_memory(
            self._namespace,
            user_id=self._active_user_id,
            kv_by_layer=kv_by_layer,
            num_tokens=len(memory_token_ids),
            token_ids=memory_token_ids,
        )
        store_write_ms = (time.perf_counter() - store_write_started) * 1000
        excluded_prep_ms = store_write_ms
        try:
            prompt = build_prompt_tokens(memory_token_ids, query_tokens)
            metrics: dict[str, Any] = {
                "kv_memory_tokens": len(memory_token_ids),
                "kv_fetch_time_ms": fetch_ms,
                "kv_compose_time_ms": compose_ms,
                "kv_store_write_time_ms": store_write_ms,
                "kv_query_tokens": len(prompt.query_token_ids),
                "kv_query_bos_stripped": int(prompt.stripped_query_bos),
                "kv_selected_chunk_ids": ordered_ids,
                "kv_block_size": cfg.engine.block_size,
                "kv_loaded_memory_tokens": loaded_memory_tokens,
                "kv_recomputed_memory_tail_tokens": recomputed_tail_tokens,
                "retrieved_chunk_count": len(hits),
            }
            output = self._engine.generate(
                prompt.prompt_token_ids,
                max_tokens=cfg.generation.max_tokens,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p,
            )
            generate_started = output.started_at
            finished = output.finished_at
            metrics["answer_generate_time_ms"] = (finished - generate_started) * 1000
            metrics["answer_total_time_ms"] = max(
                0.0, (finished - request_started) * 1000 - excluded_prep_ms
            )
            if query_started_at is not None:
                metrics["query_to_answer_ms"] = max(
                    0.0, (finished - query_started_at) * 1000 - excluded_prep_ms
                )
            prep_ms = max(0.0, (generate_started - request_started) * 1000 - excluded_prep_ms)
            metrics["kv_engine_time_to_first_token_ms"] = output.ttft_ms
            metrics["answer_time_to_first_token_ms"] = prep_ms + output.ttft_ms
            if query_started_at is not None:
                metrics["query_to_first_token_ms"] = (
                    max(0.0, (generate_started - query_started_at) * 1000 - excluded_prep_ms)
                    + output.ttft_ms
                )
            metrics["kv_store_gpu_mb"] = namespace_stats(self._namespace).get("total_gpu_mb", 0.0)
            return QueryResult(
                text=output.text,
                ttft_ms=output.ttft_ms,
                hits=list(hits),
                metrics=metrics,
            )
        finally:
            remove_user_memory(self._namespace, self._active_user_id)

    def query(self, question: str, *, top_k: int | None = None) -> QueryResult:
        """Retrieve the top-k chunks for the question and answer with their KV injected."""
        query_started = time.perf_counter()
        hits, retrieval = self.retrieve(question, top_k=top_k)
        result = self.answer(question, hits, query_started_at=query_started)
        result.retrieval = retrieval
        result.metrics.update(
            {
                "query_embedding_time_ms": retrieval.embedding_time_ms,
                "vector_db_query_time_ms": retrieval.search_time_ms,
                "query_retrieval_time_ms": retrieval.total_time_ms,
            }
        )
        return result

    def chunk_ids(self) -> list[str]:
        return list(self._chunks)

    def get_chunk(self, chunk_id: str) -> ChunkInfo | None:
        """Return metadata without exposing the token list used by future windows."""
        chunk = self._chunks.get(chunk_id)
        return replace(chunk, token_ids=list(chunk.token_ids)) if chunk is not None else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        clear_namespace(self._namespace)
        self._engine.close()
        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None
        self._corpus.close()
        self._retriever.close()
        drop_namespace(self._namespace)
        empty_cuda_cache(collect_ipc=True)

    def __enter__(self) -> "InferScale":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _ensure_encoder(self) -> ChunkedRopeEncoder:
        if self._encoder is None:
            cfg = self._config
            self._encoder = ChunkedRopeEncoder(
                model=cfg.model,
                dtype=cfg.kv.dtype,
                device=cfg.kv.device,
                max_position=cfg.kv.max_position,
            )
        return self._encoder

    def _plans_for(self, encoder: ChunkedRopeEncoder, chunks: Sequence[Chunk]) -> list[EncodingPlan]:
        cfg = self._config
        previous_count = min(cfg.context_window, len(self._chunks))
        previous = list(islice(reversed(self._chunks.values()), previous_count))
        history: deque[tuple[str, list[int]]] = deque(
            ((chunk.id, chunk.token_ids) for chunk in reversed(previous)),
            maxlen=min(cfg.context_window, len(self._chunks) + len(chunks)),
        )
        plans: list[EncodingPlan] = []
        for chunk in chunks:
            if chunk.token_ids is not None:
                token_ids = list(chunk.token_ids)
            else:
                token_ids = encode_text_no_special(encoder.tokenizer, chunk.text + cfg.prompt.chunk_separator)
            plan = build_encoding_plan(
                chunk.id,
                token_ids,
                context_token_ids=[token for _, tokens in history for token in tokens],
                context_ids=tuple(chunk_id for chunk_id, _ in history),
                max_input_tokens=cfg.kv.max_position,
            )
            plans.append(plan)
            # Prefixes never become part of the history for later chunks.
            history.append((chunk.id, plan.target_token_ids))
        return plans

    def _require_new_ids(self, chunks: Sequence[Chunk]) -> None:
        seen: set[str] = set()
        for chunk in chunks:
            if not chunk.id:
                raise ValueError("Every chunk needs a non-empty id.")
            if chunk.id in seen or chunk.id in self._chunks:
                raise ValueError(f"Duplicate chunk id: {chunk.id!r}.")
            seen.add(chunk.id)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("InferScale has been closed.")

    def _require_serving(self) -> None:
        self._require_open()
        if not self._started:
            raise RuntimeError("InferScale.start() must be called before answering queries.")
