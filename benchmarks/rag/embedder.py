from __future__ import annotations

from typing import Literal

from benchmarks.common.embedding_cache import CachedEmbedder
from inferscale.v1.embedding.openai import OpenAIEmbedder

from .config import RagBenchConfig


def build_cached_embedder(
    config: RagBenchConfig,
    mode: Literal["read", "write"],
) -> CachedEmbedder:
    if not config.embedding_cache_enabled or config.embedding_cache_dir is None:
        raise RuntimeError(
            "The RAG benchmark always runs through the embedding cache (read mode keeps "
            "answer runs offline and deterministic); keep the embedding cache enabled and "
            "configured."
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
