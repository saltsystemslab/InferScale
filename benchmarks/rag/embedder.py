from __future__ import annotations

from typing import Literal

from benchmarks.common.embedding_cache import CachedEmbedder
from inferscale.v1.embedding.local import LocalEmbedder

from .config import RagBenchConfig


def build_cached_embedder(
    config: RagBenchConfig,
    mode: Literal["read", "write"],
) -> CachedEmbedder:
    if not config.embedding_cache_enabled or config.embedding_cache_dir is None:
        raise RuntimeError(
            "The RAG benchmark always reads chunk embeddings from the embedding cache; keep "
            "the embedding cache enabled and configured."
        )
    return CachedEmbedder(
        LocalEmbedder(
            model=config.embedding_model,
            device=config.kv_device,
            batch_size=config.embed_batch_size,
        ),
        cache_dir=config.embedding_cache_dir,
        model=config.embedding_model,
        mode=mode,
    )
