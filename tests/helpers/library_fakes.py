"""Torch-free fakes for exercising the inferscale.v1 building blocks."""

from __future__ import annotations

from typing import Any

import numpy as np

from inferscale.v1.serving.vllm import GenerationOutput
from inferscale.v1.types import EncodingPlan, KVChunk, SearchHit, SearchMetrics


class FakeTokenizer:
    """Character tokenizer with a chat template that wraps message content."""

    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(char) + 2 for char in text]
        return [self.bos_token_id, *ids] if add_special_tokens else ids

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> Any:
        rendered = "".join(f"<{m['role']}>{m['content'].strip()}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        if tokenize:
            return self.encode(rendered, add_special_tokens=True)
        return rendered

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i - 2) for i in ids if i >= 2)


class FakeTensor:
    """Minimal tensor stand-in: numpy data plus the device methods the stores call."""

    def __init__(self, array: np.ndarray, device: str = "cpu") -> None:
        self.array = np.asarray(array, dtype=np.float32)
        self.device = device

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def nbytes(self) -> int:
        return int(self.array.nbytes)

    def to(self, device: Any = None, **kwargs: Any) -> "FakeTensor":
        return FakeTensor(self.array, str(device if device is not None else kwargs.get("device", self.device)))

    def contiguous(self) -> "FakeTensor":
        return self


class FakeEncoder:
    """Produces per-token fake KV so composition is a plain concatenation."""

    layers = ("model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn")

    def __init__(self, *, model: str, dtype: str, device: str, max_position: int) -> None:
        self.model = model
        self.device = device
        self.max_position = max_position
        self.tokenizer: Any = FakeTokenizer()
        self.hf_model: Any = object()
        self.released = False
        self.closed = False
        self.composed: list[list[str]] = []

    def encode(self, chunk_id: str, token_ids: list[int]) -> KVChunk:
        return KVChunk(chunk_id=chunk_id, token_ids=list(token_ids), kv_by_layer=self._kv(token_ids))

    def encode_plan(self, plan: EncodingPlan) -> KVChunk:
        return KVChunk(
            chunk_id=plan.chunk_id,
            token_ids=list(plan.target_token_ids),
            kv_by_layer=self._kv(plan.target_token_ids),
            context_ids=plan.context_ids,
            context_prefix_tokens=len(plan.context_token_ids),
            raw_context_prefix_tokens=plan.raw_context_tokens,
            context_prefix_truncated_tokens=plan.context_truncated_tokens,
        )

    def compose(self, chunks: list[KVChunk]) -> dict[str, Any]:
        self.composed.append([chunk.chunk_id for chunk in chunks])
        return {
            layer: FakeTensor(np.concatenate([chunk.kv_by_layer[layer].array for chunk in chunks], axis=1))
            for layer in self.layers
        }

    def release_model(self) -> None:
        self.released = True
        self.hf_model = None

    def close(self) -> None:
        self.closed = True

    def _kv(self, token_ids: list[int]) -> dict[str, Any]:
        tokens = np.asarray(token_ids, dtype=np.float32).reshape(1, -1, 1, 1)
        return {layer: FakeTensor(np.repeat(tokens, 2, axis=0)) for layer in self.layers}


class FakeEmbedder:
    dim = 8

    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for index, char in enumerate(text.lower()):
            vector[(ord(char) + index) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector)) or 1.0
        return (vector / norm).tolist()


class FakeIndex:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vectors: list[np.ndarray] = []
        self.payloads: list[dict[str, Any]] = []
        self.finalized = False
        self.closed = False

    @property
    def vector_count(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int | None:
        return int(self.vectors[0].shape[0]) if self.vectors else None

    def add_many(self, vectors: Any, payloads: Any, ids: Any = None) -> list[str]:
        vector_list = [np.asarray(vector, dtype=np.float32) for vector in vectors]
        payload_list = list(payloads)
        id_list = list(ids) if ids is not None else [str(index) for index in range(len(vector_list))]
        self.ids.extend(id_list)
        self.vectors.extend(vector_list)
        self.payloads.extend(dict(payload) for payload in payload_list)
        self.finalized = False
        return id_list

    def finalize(self) -> None:
        self.finalized = True

    def search(self, query_vector: Any, top_k: int, filters: Any = None) -> tuple[list[SearchHit], SearchMetrics]:
        query = np.asarray(query_vector, dtype=np.float32)
        scores = [float(vector @ query) for vector in self.vectors]
        order = sorted(range(len(scores)), key=lambda index: -scores[index])[:top_k]
        hits = [
            SearchHit(id=self.ids[index], payload=dict(self.payloads[index]), score=scores[index], distance=-scores[index], rank=rank)
            for rank, index in enumerate(order, start=1)
        ]
        return hits, SearchMetrics(search_time_ms=0.5, vector_backend="fake")

    def get(self, item_id: str) -> SearchHit | None:
        if item_id not in self.ids:
            return None
        index = self.ids.index(item_id)
        return SearchHit(id=item_id, payload=dict(self.payloads[index]), score=1.0, distance=0.0, rank=1)

    def close(self) -> None:
        self.closed = True


class FakeRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [type("Completion", (), {"text": text})()]
        self.metrics = type("Metrics", (), {"first_token_latency": 0.012})()


class FakeEngine:
    def __init__(self) -> None:
        self.tokenizer_instance = FakeTokenizer()
        self.started_with: dict[str, Any] | None = None
        self.started = False
        self.closed = False
        self.prompts: list[list[int]] = []

    @property
    def tokenizer(self) -> Any:
        if not self.started:
            raise RuntimeError("engine not started")
        return self.tokenizer_instance

    def start(self, *, kv_transfer_config: Any = None) -> float:
        self.started = True
        self.started_with = dict(kv_transfer_config) if kv_transfer_config is not None else None
        return 0.0

    def generate(self, prompt_token_ids: Any, *, max_tokens: int, temperature: float, top_p: float) -> GenerationOutput:
        prompt = list(prompt_token_ids)
        self.prompts.append(prompt)
        return GenerationOutput(
            text=f"answer over {len(prompt)} prompt tokens",
            ttft_ms=12.0,
            started_at=100.0,
            finished_at=100.25,
            raw=FakeRequestOutput("answer"),
        )

    def close(self) -> None:
        self.closed = True
