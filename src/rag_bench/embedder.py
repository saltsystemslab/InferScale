from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from locomo_jasper_bench.embedding.cache import CachedEmbedder

from .config import RagBenchConfig

# Cache-key purposes follow the repo convention (mem0 uses "add" for stored
# texts and "search" for queries), so RAG and LoCoMo runs share one
# content-addressed embedding cache without collisions.
CHUNK_EMBED_PURPOSE = "add"
QUERY_EMBED_PURPOSE = "search"


class OpenAIEmbedder:
    """Thin OpenAI embeddings client (no Mem0), wrapped by CachedEmbedder."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"max_retries": 5}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self._client = OpenAI(**kwargs)
        self._model = model

    def embed(self, text: str, *args: Any, **kwargs: Any) -> list[float]:
        return self.embed_batch([text], *args, **kwargs)[0]

    def embed_batch(self, texts: Sequence[str], *args: Any, **kwargs: Any) -> list[list[float]]:
        del args, kwargs  # purposes only shape cache keys, not the API request
        text_list = [str(text) for text in texts]
        if not text_list:
            return []
        response = self._client.embeddings.create(model=self._model, input=text_list)
        rows = sorted(response.data, key=lambda item: item.index)
        if len(rows) != len(text_list):
            raise RuntimeError(
                f"Embedding API returned {len(rows)} vectors for {len(text_list)} inputs."
            )
        return [list(row.embedding) for row in rows]


def build_cached_embedder(
    config: RagBenchConfig,
    mode: Literal["read", "write"],
) -> CachedEmbedder:
    if not config.embedding_cache_enabled:
        raise RuntimeError(
            "The RAG benchmark always runs through the embedding cache (read mode keeps "
            "answer runs offline and deterministic); remove --no-embedding-cache."
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
