from __future__ import annotations

import threading
from typing import Any

from mem0.embeddings.base import EmbeddingBase

from inferscale.v1.embedding.local import LocalEmbedder

_loaded_embedders: dict[tuple[str, str], LocalEmbedder] = {}
_loading = threading.Lock()


def shared_local_embedder(model: str, device: str) -> LocalEmbedder:
    with _loading:
        if (model, device) not in _loaded_embedders:
            _loaded_embedders[model, device] = LocalEmbedder(model=model, device=device)
        return _loaded_embedders[model, device]


class Mem0LocalEmbedder(EmbeddingBase):
    def __init__(self, config: Any = None) -> None:
        super().__init__(config)
        device = self.config.model_kwargs["device"]
        self._embedder = shared_local_embedder(self.config.model, device)
        self.config.embedding_dims = self._embedder.dim

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self._embedder.embed_documents([text])[0]

    def embed_batch(self, texts: list[str], memory_action: str = "add") -> list[list[float]]:
        return self._embedder.embed_documents(texts)
