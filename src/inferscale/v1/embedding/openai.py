"""OpenAI embeddings client."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class OpenAIEmbedder:
    """Embeds documents and queries with the OpenAI embeddings API.

    The API key comes from the OpenAI SDK's environment lookup
    (``OPENAI_API_KEY``) unless passed explicitly.
    """

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

    @property
    def model(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
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

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
