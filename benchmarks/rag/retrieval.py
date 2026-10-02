from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from loguru import logger

from inferscale.v1.index.jasper import JasperIndex, JasperIndexConfig
from inferscale.v1.retrieval import Retriever
from inferscale.v1.types import RetrievalMetrics, SearchHit

from benchmarks.common.embedding_cache import CachedEmbedder, CachedEmbeddingMissingError

from .config import RagBenchConfig
from .data_types import RagChunk


class CorpusRetriever:
    """One shared Jasper GPU index over every corpus chunk.

    Unlike LoCoMo's per-sample stores, the corpus is shared by all queries, so
    the index is built once per run and searched thousands of times. The
    read-mode embedding cache stands in for the live embedder, so answer runs
    never call the embedding API.
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
        self._index = JasperIndex(
            JasperIndexConfig(
                n_neighbors=config.jasper_n_neighbors,
                alpha=config.jasper_alpha,
                workspace_budget=config.jasper_workspace_budget,
                beam_width=config.jasper_beam_width,
            )
        )
        self._embedder = embedder
        self._retriever = Retriever(embedder, self._index, batch_size=config.embed_batch_size)
        self._built = False

    def build(self) -> dict[str, Any]:
        if self._built:
            raise RuntimeError("CorpusRetriever.build() was already called.")
        if not self._chunks:
            raise RuntimeError("Cannot build a retriever over an empty corpus.")
        import time

        started = time.perf_counter()
        payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
            }
            for chunk in self._chunks
        ]
        try:
            self._retriever.add(
                [chunk.chunk_id for chunk in self._chunks],
                [chunk.text for chunk in self._chunks],
                payloads,
            )
        except CachedEmbeddingMissingError as exc:
            raise CachedEmbeddingMissingError(self._preembed_hint(exc)) from exc
        self._retriever.finalize()
        self._built = True
        build_time_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Built Jasper corpus index chunks={} dim={} in {:.0f} ms",
            self._index.vector_count,
            self._index.dim,
            build_time_ms,
        )
        return {
            "vector_index_build_time_ms": build_time_ms,
            "embedding_cache": self._embedder.stats(),
            **self._index.memory_stats(),
        }

    def search(self, question: str, *, top_k: int) -> tuple[list[SearchHit], RetrievalMetrics]:
        if not self._built:
            raise RuntimeError("CorpusRetriever.build() must run before search().")
        try:
            return self._retriever.search(question, top_k=top_k)
        except CachedEmbeddingMissingError as exc:
            raise CachedEmbeddingMissingError(self._preembed_hint(exc)) from exc

    def close(self) -> None:
        self._retriever.close()

    def _preembed_hint(self, exc: Exception) -> str:
        return (
            f"{exc} For the RAG benchmark, warm the cache with the preembed stage for "
            f"dataset {self._config.dataset_name} and model {self._config.model} "
            f"(chunk size {self._config.chunk_size})."
        )
