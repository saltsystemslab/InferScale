from __future__ import annotations

import time
from importlib.metadata import version

from inferscale.v1.embedding.local import inference_dtype_name, last_token_embeddings

TOKENIZE_BATCH_SIZE = 256
UNIT_LENGTH_TOLERANCE = 1e-4


class QwenEmbedder:
    def __init__(
        self, *, model: str, revision: str, device: str, dtype: str, attention: str, max_length: int
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA unavailable; run on the GPU host or use --device cpu")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
        resolved_dtype = inference_dtype_name(self.device) if dtype == "auto" else dtype
        self.max_length = max_length
        load_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.model = (
            AutoModel.from_pretrained(
                model,
                revision=revision,
                torch_dtype=getattr(torch, resolved_dtype),
                attn_implementation=attention,
            )
            .to(self.device)
            .eval()
        )
        self.synchronize()
        self.metadata = {
            "model": model,
            "revision": revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": str(self.device),
            "dtype": resolved_dtype,
            "attention": attention,
            "max_length": max_length,
            "embedding_dimensions": self.model.config.hidden_size,
            "load_seconds": time.perf_counter() - load_started,
            "torch": version("torch"),
            "transformers": version("transformers"),
            "cpu_threads": torch.get_num_threads(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        }

    def token_lengths(self, texts: list[str]) -> list[int]:
        lengths: list[int] = []
        for start in range(0, len(texts), TOKENIZE_BATCH_SIZE):
            encoded = self.tokenizer(texts[start : start + TOKENIZE_BATCH_SIZE], truncation=False)
            lengths.extend(len(token_ids) for token_ids in encoded["input_ids"])
        longest = max(lengths)
        if longest > self.max_length:
            raise ValueError(
                f"Longest fact has {longest} tokens, exceeding --max-length "
                f"{self.max_length}; increase the limit to embed the complete facts"
            )
        return lengths

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def encode(self, texts: list[str]):
        return last_token_embeddings(self.tokenizer, self.model, texts)

    def validate(self, embeddings, count: int) -> None:
        if embeddings.shape != (count, self.model.config.hidden_size):
            raise ValueError(f"Invalid embedding shape: {embeddings.shape}")
        if not self.torch.isfinite(embeddings).all():
            raise ValueError("Non-finite embeddings")
        lengths = embeddings.norm(dim=1)
        unit_lengths = self.torch.ones(count)
        if not self.torch.allclose(lengths, unit_lengths, atol=UNIT_LENGTH_TOLERANCE):
            raise ValueError("Embeddings are not unit length")

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_bytes(self) -> int | None:
        if self.device.type == "cuda":
            return self.torch.cuda.max_memory_allocated(self.device)
        return None
