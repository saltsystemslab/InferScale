from __future__ import annotations

from typing import Literal

from inferscale.v1.embedding.openai import OpenAIEmbedder

from benchmarks.common.embedding_cache import CachedEmbedder

from .config import RagBenchConfig

# Cache-key purposes follow the shared convention (mem0 uses "add" for stored
# texts and "search" for queries), so RAG and LoCoMo runs share one
# content-addressed embedding cache without collisions.
CHUNK_EMBED_PURPOSE = CachedEmbedder.DOCUMENT_PURPOSE
QUERY_EMBED_PURPOSE = CachedEmbedder.QUERY_PURPOSE


def build_cached_embedder(
    config: RagBenchConfig,
    mode: Literal["read", "write"],
) -> CachedEmbedder:
    if not config.embedding_cache_enabled:
        raise RuntimeError(
            "The RAG benchmark always runs through the embedding cache (read mode keeps "
            "answer runs offline and deterministic); keep the embedding cache enabled."
        )
    return CachedEmbedder(
        OpenAIEmbedder(
            model=config.embedding_model,
            base_url=config.embedding_base_url,
            api_key=config.embedding_api_key,
        ),
        cache_dir=config.embedding_cache_dir,
        model=config.embedding_model,
        mode=mode,
        endpoint=config.embedding_base_url,
    )
