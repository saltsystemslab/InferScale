from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from loguru import logger

from locomo_jasper_bench.embedding.cache import CachedEmbedder, CachedEmbeddingMissingError
from locomo_jasper_bench.retrieval.jasper_vector_store import JasperVectorStore
from locomo_jasper_bench.runtime_paths import local_store_scratch_dir
from locomo_jasper_bench.vector_types import RetrievalMetrics, SearchHit, VectorStoreConfig

from .config import RagBenchConfig
from .data_types import RagChunk
from .embedder import CHUNK_EMBED_PURPOSE, QUERY_EMBED_PURPOSE


class CorpusRetriever:
    """One shared Jasper GPU index over every corpus chunk.

    Unlike LoCoMo's per-sample stores, the corpus is shared by all queries, so
    the index is built once per run and searched 2,556 times.
    """

    def __init__(
        self,
        *,
        config: RagBenchConfig,
        chunks: Sequence[RagChunk],
        embedder: CachedEmbedder,
    ) -> None:
        self._config = config
        self._chunks = list(chunks)
        self._embedder = embedder
        self._store = JasperVectorStore(
            local_store_scratch_dir(config.run_id) / "rag-jasper-index",
            VectorStoreConfig(
                backend="jasper",
                n_neighbors=config.jasper_n_neighbors,
                alpha=config.jasper_alpha,
                workspace_budget=config.jasper_workspace_budget,
                beam_width=config.jasper_beam_width,
            ),
        )
        self._built = False

    def build(self) -> dict[str, Any]:
        if self._built:
            raise RuntimeError("CorpusRetriever.build() was already called.")
        if not self._chunks:
            raise RuntimeError("Cannot build a retriever over an empty corpus.")
        started = time.perf_counter()
        vectors: list[list[float]] = []
        texts = [chunk.text for chunk in self._chunks]
        batch_size = self._config.embed_batch_size
        try:
            for start in range(0, len(texts), batch_size):
                vectors.extend(
                    self._embedder.embed_batch(
                        texts[start : start + batch_size], CHUNK_EMBED_PURPOSE
                    )
                )
        except CachedEmbeddingMissingError as exc:
            raise CachedEmbeddingMissingError(self._preembed_hint(exc)) from exc
        payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
            }
            for chunk in self._chunks
        ]
        self._store.add_many(vectors, payloads, ids=[chunk.chunk_id for chunk in self._chunks])
        self._store.finalize()
        self._built = True
        build_time_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Built Jasper corpus index chunks={} dim={} in {:.0f} ms",
            self._store.vector_count,
            self._store.dim,
            build_time_ms,
        )
        return {
            "vector_index_build_time_ms": build_time_ms,
            "embedding_cache": self._embedder.stats(),
            **self._store.memory_stats(),
        }

    def search(self, question: str, *, top_k: int) -> tuple[list[SearchHit], RetrievalMetrics]:
        if not self._built:
            raise RuntimeError("CorpusRetriever.build() must run before search().")
        embed_started = time.perf_counter()
        try:
            query_vector = self._embedder.embed_array(question, QUERY_EMBED_PURPOSE)
        except CachedEmbeddingMissingError as exc:
            raise CachedEmbeddingMissingError(self._preembed_hint(exc)) from exc
        embedding_time_ms = (time.perf_counter() - embed_started) * 1000
        hits, search_metrics = self._store.search(query_vector, top_k)
        return hits, RetrievalMetrics(
            embedding_time_ms=embedding_time_ms,
            search_time_ms=search_metrics.search_time_ms,
            total_time_ms=embedding_time_ms + search_metrics.search_time_ms,
            vector_backend=search_metrics.vector_backend,
            jasper_effective_beam_width=search_metrics.jasper_effective_beam_width,
        )

    def close(self) -> None:
        self._store.close()

    def _preembed_hint(self, exc: Exception) -> str:
        return (
            f"{exc} For the RAG benchmark, warm the cache with: rag-jasper-bench "
            f"--preembed-only --dataset-name {self._config.dataset_name} "
            f"--model {self._config.model} --chunk-size {self._config.chunk_size}"
        )
