from __future__ import annotations

import time
from importlib.metadata import version


class QwenEmbedder:
    """Qwen document embeddings: left padding, last-token pooling, L2 normalization."""

    def __init__(self, *, model: str, revision: str, device: str, dtype: str,
                 attention: str, max_length: int):
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
        if dtype == "auto":
            dtype = ("bfloat16" if torch.cuda.is_bf16_supported() else "float16") \
                if self.device.type == "cuda" else "float32"
        self.max_length = max_length
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, padding_side="left")
        self.model = AutoModel.from_pretrained(
            model, revision=revision, torch_dtype=getattr(torch, dtype),
            attn_implementation=attention,
        ).to(self.device).eval()
        self.synchronize()
        self.metadata = {
            "model": model, "revision": revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": str(self.device), "dtype": dtype, "attention": attention,
            "max_length": max_length, "embedding_dimensions": self.model.config.hidden_size,
            "load_seconds": time.perf_counter() - started,
            "torch": version("torch"), "transformers": version("transformers"),
            "cpu_threads": torch.get_num_threads(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        }

    def token_lengths(self, texts: list[str]) -> list[int]:
        lengths = []
        for start in range(0, len(texts), 256):
            encoded = self.tokenizer(texts[start:start + 256], truncation=False)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
        if max(lengths) > self.max_length:
            raise ValueError(
                f"Longest fact has {max(lengths)} tokens, exceeding --max-length {self.max_length}; "
                "increase the limit to embed the complete facts"
            )
        return lengths

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def encode(self, texts: list[str]):
        with self.torch.inference_mode():
            inputs = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
            inputs = inputs.to(self.device)
            hidden = self.model(**inputs, use_cache=False).last_hidden_state
            embeddings = self.torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1)
            return embeddings.cpu()

    def validate(self, embeddings, count: int) -> None:
        if embeddings.shape != (count, self.model.config.hidden_size):
            raise ValueError(f"Invalid embedding shape: {embeddings.shape}")
        if not self.torch.isfinite(embeddings).all():
            raise ValueError("Non-finite embeddings")
        if not self.torch.allclose(embeddings.norm(dim=1), self.torch.ones(count), atol=1e-4):
            raise ValueError("Embeddings are not unit length")

    def reset_peak_memory(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_bytes(self) -> int | None:
        if self.device.type == "cuda":
            return self.torch.cuda.max_memory_allocated(self.device)
        return None
