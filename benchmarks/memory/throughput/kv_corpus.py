"""The kv_injection corpus: every sample's fact chunks encoded into chunk stores before vLLM starts.

The HF encoder cannot share the GPU with vLLM, so the corpus is built first,
through the same per-sample precompute the accuracy benchmark uses (the disk
cache when enabled, otherwise encode and save), and the model weights are
released before anything else is allocated. Only the chunk stores, the
scaffold chunks, and the RoPE tables stay resident for composition.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from benchmarks.memory.composer import SampleComposer
from benchmarks.memory.data import ConversationSample
from benchmarks.memory.mem0.fact_catalog import MemoryFact, fact_catalog_hits
from benchmarks.memory.sample_precompute import SamplePrecompute
from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.projection import CorpusFootprint
from inferscale.v1.kv.encoder import TOKENIZER_PARITY_PROBE_TEXT
from inferscale.v1.kv.tokenization import encode_text_no_special
from inferscale.v1.protocols import ChunkStore


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Chunk store residency and host-to-device transfer totals for the result row."""

    host_mb: float
    h2d_bytes: int
    h2d_avg_ms: float
    h2d_p95_ms: float
    h2d_overlap_ratio: float
    staging_stall_ms: float

    @classmethod
    def from_stores(cls, stores: Sequence[ChunkStore]) -> CorpusStats:
        """Totals over the stores.

        Additive counters are summed, latency averages are weighted by transfer
        count, and the p95 is the largest store's, so with one store (the
        single-conversation workload) every figure is that store's own.
        """
        summaries = [store.get_bench_summary() for store in stores]
        transfers = [int(summary.get("total_transfers", 0)) for summary in summaries]
        total_transfers = sum(transfers)

        def weighted(key: str) -> float:
            if not total_transfers:
                return 0.0
            weighted_sum = sum(
                float(summary.get(key, 0.0)) * count for summary, count in zip(summaries, transfers)
            )
            return weighted_sum / total_transfers

        return cls(
            host_mb=sum(float(store.get_stats().get("total_host_mb", 0.0)) for store in stores),
            h2d_bytes=sum(int(summary.get("total_bytes_transferred", 0)) for summary in summaries),
            h2d_avg_ms=weighted("avg_h2d_latency_ms"),
            h2d_p95_ms=max(
                (float(summary.get("p95_h2d_latency_ms", 0.0)) for summary in summaries),
                default=0.0,
            ),
            h2d_overlap_ratio=weighted("avg_overlap_ratio"),
            staging_stall_ms=sum(
                float(summary.get("total_staging_stall_ms", 0.0)) for summary in summaries
            ),
        )


@dataclass(slots=True)
class PrecomputedCorpus:
    """Every sample's composer with its chunks in the store, plus what the later phases need."""

    composers: dict[str, SampleComposer]
    precompute: SamplePrecompute
    tokenizer: Any
    probe_token_ids: list[int]
    footprint: CorpusFootprint
    precompute_time_s: float

    def composer(self, sample_id: str) -> SampleComposer:
        return self.composers[sample_id]

    def chunk_store(self, sample_id: str) -> ChunkStore:
        store = self.composers[sample_id].chunk_store
        if store is None:
            raise RuntimeError(f"The precomputed sample {sample_id} has no chunk store.")
        return store

    def stats(self) -> CorpusStats:
        return CorpusStats.from_stores(
            [self.chunk_store(sample_id) for sample_id in self.composers]
        )

    def close(self) -> None:
        for composer in self.composers.values():
            composer.close()
        self.composers.clear()
        self.precompute.close()


def precompute_corpus(
    config: ThroughputConfig,
    samples: Sequence[ConversationSample],
    facts_by_sample: Mapping[str, tuple[MemoryFact, ...]],
) -> PrecomputedCorpus:
    """Encode or load every sample's chunks, then release the encoder weights."""
    started = time.perf_counter()
    precompute = SamplePrecompute(config)
    composers: dict[str, SampleComposer] = {}
    try:
        for number, sample in enumerate(samples, start=1):
            hits = fact_catalog_hits(facts_by_sample[sample.sample_id])
            composer, cache_hit = precompute.precompute(sample, hits)
            composers[sample.sample_id] = composer
            logger.info(
                "kv_injection: {} sample {}/{} ({} facts)",
                "loaded" if cache_hit else "encoded",
                number,
                len(samples),
                len(composer.chunks),
            )
        first = composers[samples[0].sample_id]
        # Read before the weights are released: releasing drops the encoder's tokenizer.
        tokenizer = first.encoder.tokenizer
        probe_token_ids = encode_text_no_special(tokenizer, TOKENIZER_PARITY_PROBE_TEXT)
        footprint = corpus_footprint(first)
    except BaseException:
        for composer in composers.values():
            composer.close()
        precompute.close()
        raise
    precompute.release_model()
    release_cuda_memory()
    return PrecomputedCorpus(
        composers=composers,
        precompute=precompute,
        tokenizer=tokenizer,
        probe_token_ids=probe_token_ids,
        footprint=footprint,
        precompute_time_s=time.perf_counter() - started,
    )


def corpus_footprint(composer: SampleComposer) -> CorpusFootprint:
    """The size of one sample's encoded chunks, as the projection needs it."""
    header, footer = composer.header_chunk, composer.footer_chunk
    if header is None or footer is None:
        raise RuntimeError("The precomputed sample has no scaffold chunks.")
    return CorpusFootprint(
        fact_count=len(composer.chunks),
        token_count=sum(len(chunk.token_ids) for chunk in composer.chunks.values()),
        kv_bytes=composer.stored_chunk_bytes,
        scaffold_token_count=len(header.token_ids) + len(footer.token_ids),
    )


def release_cuda_memory() -> None:
    """Return freed blocks to the driver; Jasper allocates its graphs outside torch's allocator."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
