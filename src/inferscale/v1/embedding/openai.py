from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        client_options: dict[str, Any] = {
            "max_retries": 5,
            "base_url": base_url or DEFAULT_OPENAI_BASE_URL,
        }
        if api_key:
            client_options["api_key"] = api_key
        self._client = OpenAI(**client_options)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = [str(text) for text in texts]
        if not text_list:
            return []
        response = self._client.embeddings.create(model=self._model, input=text_list)
        rows_in_input_order = sorted(response.data, key=lambda row: row.index)
        if len(rows_in_input_order) != len(text_list):
            raise RuntimeError(
                f"Embedding API returned {len(rows_in_input_order)} vectors "
                f"for {len(text_list)} inputs."
            )
        return [list(row.embedding) for row in rows_in_input_order]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
