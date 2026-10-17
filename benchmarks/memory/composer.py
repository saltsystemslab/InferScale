"""Per-sample chunk encoding and top-k composition for LoCoMo Mem0 facts.

The composer owns one LoCoMo sample's fact-chunk corpus: it encodes every
fact (with its preceding conversation turns as an encoding-only prefix),
keeps the scaffold chunks as local GPU tensors, and composes the retrieved
facts in reverse rank order with the fact-plan accounting the benchmark
records.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from loguru import logger

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
from inferscale.v1.kv.tokenization import encode_text_no_special
from inferscale.v1.protocols import ChunkStore
from inferscale.v1.serving.vllm import empty_cuda_cache
from inferscale.v1.types import KVChunk, SearchHit

from .chunk_cache import CachedSampleEncode
from .context import build_fact_context_encoding_plan, plan_kv_memory
from .data import ConversationSample
from .facts import reverse_ranked_memory_facts, unique_memory_facts
from .prompting import extract_memory_scaffold_token_ids, format_memory_fact
from .types import ComposedSampleMemory, RetrievedFact


class SampleComposer:
    """GPU-resident pre-RoPE chunk encoder and top-k composer for one sample."""

    def __init__(
        self,
        *,
        encoder: ChunkedRopeEncoder,
        context_window: int = 0,
        block_size: int = 16,
        chunk_store: ChunkStore | None = None,
    ) -> None:
        if context_window < 0:
            raise ValueError("context_window must be >= 0.")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")
        if encoder.tokenizer is None:
            raise RuntimeError(
                "Cannot create a sample composer because the HF encoder has been released."
            )
        self.encoder = encoder
        self.device = encoder.device
        self.max_position = encoder.max_position
        self.context_window = context_window
        self.block_size = block_size
        # When set, the fact-chunk corpus lives in this store after
        # move_chunks_to_store, with its ID lookup map on the GPU; self.chunks
        # then holds metadata-only chunks and compose() stages the selected
        # chunks per request. Scaffold chunks stay local GPU tensors either way.
        self.chunk_store = chunk_store
        self.chunks: dict[str, KVChunk] = {}
        self.header_chunk: KVChunk | None = None
        self.empty_memory_chunk: KVChunk | None = None
        self.footer_chunk: KVChunk | None = None
        self._chunks_in_store = False
        self._stored_chunk_bytes = 0
        self._stored_chunk_layers = 0
        self._fact_token_ids: dict[str, list[int]] = {}
        self._turn_token_ids: dict[str, list[int]] = {}
        self._facts: list[RetrievedFact] = []
        self._facts_by_id: dict[str, RetrievedFact] = {}
        self._sample: ConversationSample | None = None

    @classmethod
    def from_cached(
        cls,
        *,
        encoder: ChunkedRopeEncoder,
        cached: CachedSampleEncode,
        sample: ConversationSample,
        facts: Sequence[SearchHit],
        context_window: int = 0,
        block_size: int = 16,
        chunk_store: ChunkStore | None = None,
    ) -> SampleComposer:
        """A composer over cache-loaded chunks, skipping the encode entirely.

        The live facts are re-attached (not read from the cache) so compose()
        compares retrieved facts against the real catalog; fact token ids are
        recomputed from the tokenizer. With a chunk store, the cache-loaded
        KV moves straight into it.
        """
        composer = cls(
            encoder=encoder,
            context_window=context_window,
            block_size=block_size,
            chunk_store=chunk_store,
        )
        composer._attach_facts(sample, facts)
        composer.chunks = dict(cached.fact_chunks)
        composer.header_chunk = cached.scaffold_chunks.get("header")
        composer.empty_memory_chunk = cached.scaffold_chunks.get("empty_memory")
        composer.footer_chunk = cached.scaffold_chunks.get("footer")
        composer.move_chunks_to_store()
        return composer

    @property
    def chunks_in_store(self) -> bool:
        return self._chunks_in_store

    @property
    def stored_chunk_bytes(self) -> int:
        """KV bytes of the fact chunks as measured when they moved into the store."""
        return self._stored_chunk_bytes

    @property
    def stored_chunk_layers(self) -> int:
        return self._stored_chunk_layers

    def scaffold_chunks(self) -> dict[str, KVChunk | None]:
        """The scaffold chunks by cache slot name."""
        return {
            "header": self.header_chunk,
            "empty_memory": self.empty_memory_chunk,
            "footer": self.footer_chunk,
        }

    def encoded_scaffold_chunks(self) -> list[KVChunk]:
        return [chunk for chunk in self.scaffold_chunks().values() if chunk is not None]

    def move_chunks_to_store(self) -> None:
        """Move fact-chunk KV into the chunk store, keeping metadata locally.

        No-op without a chunk store. Callers that save the disk cache must do
        so before this move: afterwards the local chunks are metadata-only and
        the store owns the tensors until close().
        """
        if self.chunk_store is None:
            return
        if not self._chunks_in_store:
            first_chunk = next(iter(self.chunks.values()), None)
            self._stored_chunk_layers = len(first_chunk.kv_by_layer) if first_chunk else 0
            self._stored_chunk_bytes = sum(chunk_nbytes(chunk) for chunk in self.chunks.values())
            self.chunks = register_chunks(self.chunk_store, self.chunks)
            # Registration transfers ownership even if GPU finalization fails;
            # a retry must preserve these payloads instead of registering metadata.
            self._chunks_in_store = True
        finalize_chunk_store(self.chunk_store)

    def encode_sample(self, sample: ConversationSample, facts: Sequence[SearchHit]) -> None:
        if self.encoder.tokenizer is None:
            raise RuntimeError("Cannot encode a sample because the HF encoder has been released.")
        self._attach_facts(sample, facts)
        scaffold = extract_memory_scaffold_token_ids(
            self.encoder.tokenizer, block_size=self.block_size
        )
        self.header_chunk = self.encoder.encode("__header__", scaffold.header_token_ids)
        self.empty_memory_chunk = self.encoder.encode("__empty_memory__", scaffold.empty_token_ids)
        self.footer_chunk = self.encoder.encode("__footer__", scaffold.footer_token_ids)
        logger.info(
            "Pre-RoPE encoding {} unique Mem0 fact chunks for sample_id={} context_window={}",
            len(self._facts),
            sample.sample_id,
            self.context_window,
        )
        for fact in self._facts:
            self.chunks[fact.memory_id] = self._encode_fact_chunk(sample, fact)
        logger.info(
            "Pre-RoPE encoded {} chunks for sample_id={}", len(self.chunks), sample.sample_id
        )

    def compose(
        self, hits: list[SearchHit], *, memory_token_budget: int | None = None
    ) -> ComposedSampleMemory:
        """Compose the retrieved facts' KV in reverse rank order, best fact last."""
        started = time.perf_counter()
        header, empty_memory, footer = self._encoded_scaffold()
        retrieved_facts = unique_memory_facts(hits)
        selected_facts = reverse_ranked_memory_facts(hits)
        self._check_selected_facts(selected_facts)
        budget = self.max_position if memory_token_budget is None else memory_token_budget
        budget = min(self.max_position, budget)
        heading_chunks = [] if selected_facts else [empty_memory]
        heading_tokens = sum(len(chunk.token_ids) for chunk in heading_chunks)
        fact_plan = plan_kv_memory(
            selected_facts,
            retrieved_facts=retrieved_facts,
            context_window=self.context_window,
            memory_token_budget=budget,
            scaffold_token_count=len(header.token_ids) + heading_tokens + len(footer.token_ids),
            fact_token_ids=self._fact_token_ids,
            encoded_chunks=self.chunks,
        )
        if fact_plan.memory_tokens > budget:
            raise AssertionError("Planned KV memory exceeds its token budget.")
        fact_tokens_end = len(header.token_ids) + heading_tokens + fact_plan.fact_tokens
        loaded_tokens = self._block_aligned_prefix(fact_plan.memory_tokens, fact_tokens_end)
        kv_by_layer, token_ids = self._compose_chunks(
            header, heading_chunks, [fact.memory_id for fact in selected_facts], footer
        )
        if len(token_ids) != fact_plan.memory_tokens:
            raise AssertionError("Planned memory token count does not match the composed memory.")
        return ComposedSampleMemory(
            kv_by_layer=kv_by_layer,
            token_ids=token_ids,
            num_tokens=len(token_ids),
            compose_time_ms=(time.perf_counter() - started) * 1000,
            fact_plan=fact_plan,
            loaded_memory_tokens=loaded_tokens,
            recomputed_memory_tail_tokens=fact_plan.memory_tokens - loaded_tokens,
            fact_tokens_end=fact_tokens_end,
        )

    def close(self) -> None:
        if self.chunk_store is not None:
            close_chunk_store(self.chunk_store)
            self.chunk_store = None
        self.chunks = {}
        self.header_chunk = None
        self.empty_memory_chunk = None
        self.footer_chunk = None
        self._fact_token_ids = {}
        self._turn_token_ids = {}
        self._facts = []
        self._facts_by_id = {}
        self._sample = None
        empty_cuda_cache()

    def _attach_facts(self, sample: ConversationSample, facts: Sequence[SearchHit]) -> None:
        self._sample = sample
        self._facts = unique_memory_facts(facts)
        self._facts_by_id = {fact.memory_id: fact for fact in self._facts}
        self._fact_token_ids = {
            fact.memory_id: encode_text_no_special(self.encoder.tokenizer, format_memory_fact(fact))
            for fact in self._facts
        }

    def _encode_fact_chunk(self, sample: ConversationSample, fact: RetrievedFact) -> KVChunk:
        plan = build_fact_context_encoding_plan(
            fact,
            sample,
            tokenizer=self.encoder.tokenizer,
            context_window=self.context_window,
            max_input_tokens=self.max_position,
            fact_token_ids=self._fact_token_ids,
            turn_token_ids=self._turn_token_ids,
        )
        return self.encoder.encode_plan(plan)

    def _encoded_scaffold(self) -> tuple[KVChunk, KVChunk, KVChunk]:
        if self._sample is None:
            raise RuntimeError("Cannot compose KV memory before encoding a sample.")
        if (
            self.header_chunk is None
            or self.empty_memory_chunk is None
            or self.footer_chunk is None
        ):
            raise RuntimeError("Cannot compose KV memory before encoding its prompt scaffold.")
        return self.header_chunk, self.empty_memory_chunk, self.footer_chunk

    def _check_selected_facts(self, selected_facts: Sequence[RetrievedFact]) -> None:
        """Every retrieved fact must be the fact that was encoded, unchanged."""
        for fact in selected_facts:
            encoded = self._facts_by_id.get(fact.memory_id)
            if encoded is None:
                raise RuntimeError(
                    f"Retrieved Mem0 fact was not present in the precompute catalog: "
                    f"{fact.memory_id}."
                )
            if encoded != fact:
                raise RuntimeError(
                    f"Retrieved Mem0 fact changed after precompute: {fact.memory_id}."
                )

    def _block_aligned_prefix(self, memory_tokens: int, fact_tokens_end: int) -> int:
        """Tokens vLLM loads from the injected KV; the rest of the memory is recomputed."""
        loaded_tokens = (memory_tokens // self.block_size) * self.block_size
        if loaded_tokens < fact_tokens_end:
            raise RuntimeError(
                "The block-aligned KV prefix would leave retrieved fact tokens in the "
                "recomputed tail: "
                f"loaded={loaded_tokens} facts_end={fact_tokens_end} "
                f"memory={memory_tokens} block_size={self.block_size}."
            )
        return loaded_tokens

    def _compose_chunks(
        self,
        header: KVChunk,
        heading_chunks: list[KVChunk],
        selected_ids: list[str],
        footer: KVChunk,
    ) -> tuple[dict[str, Any], list[int]]:
        store = self._ensure_store()
        # CPU payloads are staged after the GPU map selects their rows; the
        # staging tensors stay valid through composition.
        selected = fetch_chunks(store, self.chunks, selected_ids)
        chunks = [header, *heading_chunks, *selected, footer]
        try:
            kv_by_layer = self.encoder.compose(chunks)
        finally:
            release_chunks(store, selected_ids)
        return kv_by_layer, [token for chunk in chunks for token in chunk.token_ids]

    def _ensure_store(self) -> ChunkStore:
        if self.chunk_store is None:
            # A composer built without a store (offline cache writes) still
            # serves compositions through the GPU lookup map.
            self.chunk_store = build_chunk_store(
                "gpu", device=str(self.device), top_k=max(1, len(self.chunks))
            )
        self.move_chunks_to_store()
        return self.chunk_store
