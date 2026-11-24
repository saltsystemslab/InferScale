from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
WARMUP_TEXT = "warm-up"


class LocalEmbedder:
    def __init__(
        self,
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cuda:0",
        batch_size: int = 128,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self._model_name = model
        self._batch_size = batch_size
        self._one_batch_at_a_time = threading.Lock()
        target_device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
        self._model = (
            AutoModel.from_pretrained(model, torch_dtype=_inference_dtype(torch, target_device))
            .to(target_device)
            .eval()
        )
        self.embed_query(WARMUP_TEXT)

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return int(self._model.config.hidden_size)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        text_list = [str(text) for text in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(text_list), self._batch_size):
            vectors.extend(self._encode(text_list[start : start + self._batch_size]).tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._encode([str(text)])[0].tolist()

    def _encode(self, texts: list[str]) -> Any:
        with self._one_batch_at_a_time:
            return last_token_embeddings(self._tokenizer, self._model, texts)


def last_token_embeddings(tokenizer: Any, model: Any, texts: list[str]) -> Any:
    import torch

    with torch.inference_mode():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings,
            return_tensors="pt",
        ).to(model.device)
        hidden = model(**inputs, use_cache=False).last_hidden_state
        return torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1).cpu()


def _inference_dtype(torch: Any, device: Any) -> Any:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
