"""Per-sample chunk encoding and top-k composition for LoCoMo Mem0 facts.

The composer owns one LoCoMo sample's fact-chunk corpus: it encodes every
fact (with its preceding conversation turns as an encoding-only prefix),
keeps the scaffold chunks as local GPU tensors, and composes the retrieved
facts in reverse rank order with the fact-plan accounting the benchmark
records.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from typing import Any

from inferscale.v1.kv.chunk_store import (
    build_chunk_store,
    chunk_nbytes,
    close_chunk_store,
    fetch_chunks,
    finalize_chunk_store,
    register_chunks,
    release_chunks,
)
from inferscale.v1.kv.encoder import ChunkedRopeEncoder
from inferscale.v1.kv.memory_store import bytes_to_mb
from inferscale.v1.kv.tokenization import encode_text_no_special
from inferscale.v1.types import KVChunk, SearchHit

from .chunk_cache import CachedSampleEncode
from .context import (
    build_fact_context_encoding_plan,
    build_memory_fact_plan,
    reverse_ranked_memory_facts,
    unique_memory_facts,
)
from .data import ConversationSample
from .prompting import extract_memory_scaffold_token_ids, format_memory_fact
from .types import ComposedMemory, MemoryFact

logger = logging.getLogger(__name__)


class SampleComposer:
    """GPU-resident pre-RoPE chunk encoder and top-k composer for one sample."""

    def __init__(
        self,
        *,
        encoder: ChunkedRopeEncoder,
        context_window: int = 0,
        block_size: int = 16,
        chunk_store: Any | None = None,
    ) -> None:
        if context_window < 0:
            raise ValueError("context_window must be >= 0.")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")

        self.encoder = encoder
        self.model = encoder.model
        self.device = encoder.device
        self.max_position = encoder.max_position
        self.context_window = context_window
        self.block_size = block_size
        # When set, the fact-chunk corpus lives in this backend store
        # after move_chunks_to_store, with its ID lookup map on GPU;
        # self.chunks then holds metadata-only chunks and compose() stages
        # the selected chunks per request. Scaffold chunks stay local GPU
        # tensors either way.
        self.chunk_store = chunk_store
        self.chunks: dict[str, KVChunk] = {}
        self._chunks_in_store = False
        self._stored_chunk_bytes = 0
        self._stored_chunk_layers = 0
        self._fact_token_ids: dict[str, list[int]] = {}
        self._turn_token_ids: dict[str, list[int]] = {}
        self._facts: list[MemoryFact] = []
        self._facts_by_id: dict[str, MemoryFact] = {}
        self._sample: ConversationSample | None = None
        self.header_chunk: KVChunk | None = None
        self.empty_memory_chunk: KVChunk | None = None
        self.footer_chunk: KVChunk | None = None
        if encoder.tokenizer is None:
            raise RuntimeError("Cannot create sample composer because the HF encoder has been released.")

    @classmethod
    def from_cached(
        cls,
        *,
        encoder: ChunkedRopeEncoder,
        cached: "CachedSampleEncode",
        sample: ConversationSample,
        facts: Sequence[SearchHit],
        context_window: int = 0,
        block_size: int = 16,
        chunk_store: Any | None = None,
    ) -> "SampleComposer":
        """Composer over cache-loaded chunks, skipping the encode entirely.

        The live facts are re-attached (not read from the cache) so
        compose()'s retrieved-vs-precomputed equality checks compare against
        the real catalog; fact token ids are recomputed from the tokenizer.
        With a chunk_store, the cache-loaded KV moves straight into it.
        """
        composer = cls(
            encoder=encoder,
            context_window=context_window,
            block_size=block_size,
            chunk_store=chunk_store,
        )
        composer._sample = sample
        composer._facts = unique_memory_facts(facts)
        composer._facts_by_id = {fact.memory_id: fact for fact in composer._facts}
        composer.chunks = dict(cached.fact_chunks)
        composer.header_chunk = cached.scaffold_chunks.get("header")
        composer.empty_memory_chunk = cached.scaffold_chunks.get("empty_memory")
        composer.footer_chunk = cached.scaffold_chunks.get("footer")
        composer._fact_token_ids = {
            fact.memory_id: encode_text_no_special(
                encoder.tokenizer,
                format_memory_fact(fact),
            )
            for fact in composer._facts
        }
        composer.move_chunks_to_store()
        return composer

    def move_chunks_to_store(self) -> None:
        """Move fact-chunk KV into the chunk store, keeping metadata locally.

        No-op without a chunk store. Callers that save the disk cache must
        do so BEFORE this move: afterwards the local chunks are
        metadata-only and the store owns the tensors until close().
        """
        if self.chunk_store is None:
            return
        if not self._chunks_in_store:
            first_chunk = next(iter(self.chunks.values()), None)
            self._stored_chunk_layers = len(first_chunk.kv_by_layer) if first_chunk else 0
            self._stored_chunk_bytes = sum(chunk_nbytes(chunk) for chunk in self.chunks.values())
            self.chunks = register_chunks(self.chunk_store, self.chunks)
            # Registration transfers ownership even if GPU finalization fails.
            # A retry must preserve these payloads instead of registering metadata.
            self._chunks_in_store = True
        finalize_chunk_store(self.chunk_store)

    def encode_sample(
        self,
        sample: ConversationSample,
        facts: Sequence[SearchHit],
    ) -> None:
        if self.encoder.tokenizer is None:
            raise RuntimeError("Cannot encode sample because the HF encoder has been released.")
        self._sample = sample
        self._facts = unique_memory_facts(facts)
        self._facts_by_id = {fact.memory_id: fact for fact in self._facts}
        scaffold = extract_memory_scaffold_token_ids(
            self.encoder.tokenizer,
            block_size=self.block_size,
        )
        self.header_chunk = self.encoder.encode("__header__", scaffold.header_token_ids)
        self.empty_memory_chunk = self.encoder.encode("__empty_memory__", scaffold.empty_token_ids)
        self.footer_chunk = self.encoder.encode("__footer__", scaffold.footer_token_ids)
        self._fact_token_ids = {
            fact.memory_id: encode_text_no_special(
                self.encoder.tokenizer,
                format_memory_fact(fact),
            )
            for fact in self._facts
        }
        logger.info(
            "Pre-RoPE encoding %d unique Mem0 fact chunks for sample_id=%s context_window=%d",
            len(self._facts),
            sample.sample_id,
            self.context_window,
        )
        for fact in self._facts:
            self.chunks[fact.memory_id] = self._encode_fact_chunk(fact)
        logger.info("Pre-RoPE encoded %d chunks for sample_id=%s", len(self.chunks), sample.sample_id)

    def compose(
        self,
        hits: list[SearchHit],
        *,
        memory_token_budget: int | None = None,
    ) -> ComposedMemory:
        started = time.perf_counter()
        if self._sample is None:
            raise RuntimeError("Cannot compose KV memory before encoding a sample.")
        if (
            self.header_chunk is None
            or self.empty_memory_chunk is None
            or self.footer_chunk is None
        ):
            raise RuntimeError("Cannot compose KV memory before encoding its prompt scaffold.")
        retrieved_facts = unique_memory_facts(hits)
        selected_facts = reverse_ranked_memory_facts(hits)
        for fact in selected_facts:
            cached = self._facts_by_id.get(fact.memory_id)
            if cached is None:
                raise RuntimeError(
                    f"Retrieved Mem0 fact was not present in the precompute catalog: {fact.memory_id}."
                )
            if cached != fact:
                raise RuntimeError(
                    f"Retrieved Mem0 fact changed after precompute: {fact.memory_id}."
                )

        effective_memory_token_budget = min(
            self.max_position,
            self.max_position if memory_token_budget is None else memory_token_budget,
        )
        memory_heading_chunks = [] if selected_facts else [self.empty_memory_chunk]
        memory_heading_tokens = sum(
            len(chunk.token_ids) for chunk in memory_heading_chunks
        )
        scaffold_token_count = (
            len(self.header_chunk.token_ids)
            + memory_heading_tokens
            + len(self.footer_chunk.token_ids)
        )
        fact_plan = build_memory_fact_plan(
            selected_facts,
            retrieved_facts=retrieved_facts,
            context_window=self.context_window,
            memory_token_budget=effective_memory_token_budget,
            scaffold_token_count=scaffold_token_count,
            fact_token_ids=self._fact_token_ids,
            encoded_chunks=self.chunks,
        )
        if fact_plan.memory_tokens > effective_memory_token_budget:
            raise AssertionError("Planned KV memory exceeds its token budget.")

        fact_tokens_end = (
            len(self.header_chunk.token_ids)
            + memory_heading_tokens
            + fact_plan.fact_tokens
        )
        loaded_memory_tokens = (
            fact_plan.memory_tokens // self.block_size
        ) * self.block_size
        recomputed_memory_tail_tokens = fact_plan.memory_tokens - loaded_memory_tokens
        if loaded_memory_tokens < fact_tokens_end:
            raise RuntimeError(
                "The block-aligned KV prefix would leave retrieved fact tokens in the "
                "recomputed tail: "
                f"loaded={loaded_memory_tokens} facts_end={fact_tokens_end} "
                f"memory={fact_plan.memory_tokens} block_size={self.block_size}."
            )

        selected_ids = [fact.memory_id for fact in selected_facts]
        if self.chunk_store is None:
            # Standalone composers may retain tensors for offline cache writes,
            # but serving always resolves IDs through the GPU lookup map.
            self.chunk_store = build_chunk_store(
                "gpu", device=str(self.device), top_k=max(1, len(self.chunks))
            )
        self.move_chunks_to_store()
        # CPU payloads are staged after the GPU map selects their rows.
        # Staging tensors are record_stream-protected through composition.
        selected = fetch_chunks(self.chunk_store, self.chunks, selected_ids)

        chunks = [self.header_chunk, *memory_heading_chunks, *selected]
        chunks.append(self.footer_chunk)
        try:
            kv_by_layer = self.encoder.compose(chunks)
        finally:
            release_chunks(self.chunk_store, selected_ids)
        token_ids: list[int] = []
        for chunk in chunks:
            token_ids.extend(chunk.token_ids)
        if len(token_ids) != fact_plan.memory_tokens:
            raise AssertionError(
                "Planned memory token count does not match the composed KV memory."
            )
        return ComposedMemory(
            kv_by_layer=kv_by_layer,
            token_ids=token_ids,
            num_tokens=len(token_ids),
            compose_time_ms=(time.perf_counter() - started) * 1000,
            fact_plan=fact_plan,
            loaded_memory_tokens=loaded_memory_tokens,
            recomputed_memory_tail_tokens=recomputed_memory_tail_tokens,
            fact_tokens_end=fact_tokens_end,
        )

    def cache_stats(self) -> dict[str, Any]:
        scaffold_chunks = [
            chunk
            for chunk in (
                getattr(self, "header_chunk", None),
                getattr(self, "empty_memory_chunk", None),
                getattr(self, "footer_chunk", None),
            )
            if chunk is not None
        ]
        fact_chunks_by_id = getattr(self, "chunks", {}) or {}
        fact_chunks = list(fact_chunks_by_id.values())
        chunks = list(scaffold_chunks)
        chunks.extend(fact_chunks)

        total_tokens = 0
        layer_count = 0
        devices: set[str] = set()
        for chunk in chunks:
            total_tokens += len(chunk.token_ids)
            layer_count = max(layer_count, len(chunk.kv_by_layer))
            for tensor in chunk.kv_by_layer.values():
                device = getattr(tensor, "device", None)
                if device is not None:
                    devices.add(str(device))
        prefix_tensor_bytes = sum(chunk_nbytes(chunk) for chunk in scaffold_chunks)
        if self._chunks_in_store:
            # The fact-chunk KV lives in the chunk store and the local
            # chunks are metadata-only; report the sizes captured at move
            # time and the store's residency.
            fact_chunk_tensor_bytes = self._stored_chunk_bytes
            layer_count = max(layer_count, self._stored_chunk_layers)
            residency = "cpu" if getattr(self.chunk_store, "num_staging_slots", 0) else "gpu"
            devices.add("cpu(pinned)" if residency == "cpu" else str(self.device))
        else:
            fact_chunk_tensor_bytes = sum(chunk_nbytes(chunk) for chunk in fact_chunks)
            residency = "gpu"
        total_bytes = prefix_tensor_bytes + fact_chunk_tensor_bytes
        chunk_metadata_cpu_bytes = _chunk_metadata_cpu_bytes(fact_chunks_by_id)
        store_stats = self.chunk_store.get_stats() if self.chunk_store is not None else {}
        chunk_map_gpu_bytes = int(store_stats.get("chunk_map_bytes", 0))

        stats = {
            "kv_chunk_cache_residency": residency,
            "kv_precomputed_chunks": max(0, len(chunks) - len(scaffold_chunks)),
            "kv_precomputed_chunks_with_prefix": len(chunks),
            "kv_precomputed_tokens": total_tokens,
            "kv_precomputed_layers": layer_count,
            "kv_precomputed_gpu_mb": total_bytes / (1024 * 1024),
            "kv_precomputed_devices": ",".join(sorted(devices)),
            "llama_kv_chunk_count": len(fact_chunks),
            "llama_kv_chunk_metadata_cpu_bytes": chunk_metadata_cpu_bytes,
            "llama_kv_chunk_metadata_cpu_mb": bytes_to_mb(chunk_metadata_cpu_bytes),
            "llama_kv_chunk_map_gpu_bytes": chunk_map_gpu_bytes,
            "llama_kv_chunk_map_gpu_mb": bytes_to_mb(chunk_map_gpu_bytes),
            "llama_kv_chunk_tensor_gpu_bytes": fact_chunk_tensor_bytes,
            "llama_kv_chunk_tensor_gpu_mb": bytes_to_mb(fact_chunk_tensor_bytes),
            "llama_kv_prefix_tensor_gpu_bytes": prefix_tensor_bytes,
            "llama_kv_prefix_tensor_gpu_mb": bytes_to_mb(prefix_tensor_bytes),
            "llama_kv_total_tensor_gpu_bytes": total_bytes,
            "llama_kv_total_tensor_gpu_mb": bytes_to_mb(total_bytes),
        }
        for key, value in store_stats.items():
            stats[f"kv_chunk_store_{key}"] = value
        return stats

    def close(self) -> None:
        import gc
        import torch

        chunk_store = getattr(self, "chunk_store", None)
        if chunk_store is not None:
            close_chunk_store(chunk_store)
        for attr in (
            "encoder",
            "chunk_store",
            "chunks",
            "header_chunk",
            "empty_memory_chunk",
            "footer_chunk",
            "_fact_token_ids",
            "_turn_token_ids",
            "_facts",
            "_facts_by_id",
            "_sample",
        ):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()

    def _encode_fact_chunk(self, fact: MemoryFact) -> KVChunk:
        if self._sample is None:
            raise RuntimeError("Cannot encode a fact chunk before encode_sample().")
        plan = build_fact_context_encoding_plan(
            fact,
            self._sample,
            tokenizer=self.encoder.tokenizer,
            context_window=self.context_window,
            max_input_tokens=self.max_position,
            fact_token_ids=self._fact_token_ids,
            turn_token_ids=self._turn_token_ids,
        )
        return self.encoder.encode_plan(plan)


def _chunk_metadata_cpu_bytes(chunks_by_id: dict[str, KVChunk]) -> int:
    total = sys.getsizeof(chunks_by_id)
    for chunk_id, chunk in chunks_by_id.items():
        total += sys.getsizeof(chunk_id)
        total += _chunk_cpu_bytes(chunk)
    return total


def _chunk_cpu_bytes(chunk: KVChunk) -> int:
    total = sys.getsizeof(chunk)
    total += sys.getsizeof(chunk.token_ids)
    total += sum(sys.getsizeof(token_id) for token_id in chunk.token_ids)
    total += sys.getsizeof(chunk.kv_by_layer)
    total += sum(sys.getsizeof(layer_name) for layer_name in chunk.kv_by_layer)
    return total
