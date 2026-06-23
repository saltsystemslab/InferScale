from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any

from ..data import ConversationSample, Turn, format_turn_for_memory
from ..vector_types import SearchHit
from .prompting import MEMORY_PREFIX_TEXT, selected_turn_ids
from .submodule import require_ai_memory_submodule

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EncodedChunk:
    turn_id: str
    token_ids: list[int]
    kv_by_layer: dict[str, Any]
    text: str


@dataclass(slots=True)
class ComposedMemory:
    kv_by_layer: dict[str, Any]
    token_ids: list[int]
    num_tokens: int
    compose_time_ms: float
    selected_turn_ids: list[str]


def torch_dtype(dtype_name: str) -> Any:
    require_ai_memory_submodule()
    import torch

    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported KV dtype: {dtype_name!r}")


class ChunkedRopeSampleComposer:
    """GPU-resident pre-RoPE chunk encoder and top-k composer for one sample."""

    def __init__(
        self,
        *,
        model: str,
        dtype: str,
        device: str,
        max_position: int,
    ) -> None:
        require_ai_memory_submodule()
        import torch
        from encode_memories_pre_rope import PreRoPEMemoryEncoder
        from rope_inject import extract_cos_sin_from_model

        if not torch.cuda.is_available():
            raise RuntimeError("KV injection requires a CUDA device.")

        self.model = model
        self.device = device
        self.max_position = max_position
        self.encoder = PreRoPEMemoryEncoder(
            model_name=model,
            dtype=torch_dtype(dtype),
            device=device,
        )
        self.encoder.load_model()
        self.tokenizer = self.encoder._tokenizer
        self.hf_model = self.encoder._model
        if self.tokenizer is None or self.hf_model is None:
            raise RuntimeError("Pre-RoPE encoder did not load a model/tokenizer.")

        cfg = self.hf_model.config
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        positions = torch.arange(max_position, device=device)
        self.cos_table, self.sin_table = extract_cos_sin_from_model(
            self.hf_model,
            positions,
            self.head_dim,
        )
        self.chunks: dict[str, EncodedChunk] = {}
        self.prefix_chunk = self._encode_text_chunk("__prefix__", MEMORY_PREFIX_TEXT)

    def encode_sample(self, sample: ConversationSample, turn_ids: set[str] | None = None) -> None:
        turns = sample.turns
        if turn_ids is not None:
            turns = [turn for turn in sample.turns if turn.id in turn_ids]
        logger.info("Pre-RoPE encoding %d memory chunks for sample_id=%s", len(turns), sample.sample_id)
        for turn in turns:
            self.chunks[turn.id] = self._encode_turn_chunk(turn)
        logger.info("Pre-RoPE encoded %d chunks for sample_id=%s", len(self.chunks), sample.sample_id)

    def compose(self, hits: list[SearchHit]) -> ComposedMemory:
        started = time.perf_counter()
        turn_ids = selected_turn_ids(hits)
        if not turn_ids:
            raise RuntimeError("Cannot compose KV memory because retrieval returned no turn ids.")

        selected = []
        missing = []
        for turn_id in turn_ids:
            chunk = self.chunks.get(turn_id)
            if chunk is None:
                missing.append(turn_id)
            else:
                selected.append(chunk)
        if missing:
            raise RuntimeError(
                "Retrieved memory chunks were not pre-encoded: " + ", ".join(missing[:5])
            )

        chunks = [self.prefix_chunk, *selected]
        kv_by_layer = self._compose_chunks(chunks)
        token_ids: list[int] = []
        for chunk in chunks:
            token_ids.extend(chunk.token_ids)
        if len(token_ids) > self.max_position:
            raise RuntimeError(
                f"Composed memory has {len(token_ids)} tokens, exceeding kv_max_position={self.max_position}."
            )
        return ComposedMemory(
            kv_by_layer=kv_by_layer,
            token_ids=token_ids,
            num_tokens=len(token_ids),
            compose_time_ms=(time.perf_counter() - started) * 1000,
            selected_turn_ids=turn_ids,
        )

    def close(self) -> None:
        import gc
        import torch

        for attr in ("encoder", "hf_model", "tokenizer", "cos_table", "sin_table", "chunks", "prefix_chunk"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()

    def _encode_turn_chunk(self, turn: Turn) -> EncodedChunk:
        return self._encode_text_chunk(turn.id, format_turn_for_memory(turn).strip() + "\n")

    def _encode_text_chunk(self, turn_id: str, text: str) -> EncodedChunk:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"Memory chunk tokenized to zero tokens: {turn_id}")
        return EncodedChunk(
            turn_id=turn_id,
            token_ids=token_ids,
            kv_by_layer=_encode_token_chunk_pre_rope(self.hf_model, token_ids),
            text=text,
        )

    def _compose_chunks(self, chunks: list[EncodedChunk]) -> dict[str, Any]:
        from rope_inject import rotate_chunk_at_virtual_position
        import torch

        layer_names = list(chunks[0].kv_by_layer.keys())
        composed: dict[str, Any] = {}
        total_tokens = sum(len(chunk.token_ids) for chunk in chunks)
        if total_tokens > self.max_position:
            raise RuntimeError(
                f"Composed memory has {total_tokens} tokens, exceeding kv_max_position={self.max_position}."
            )

        for layer_name in layer_names:
            rotated_keys = []
            values = []
            virtual_pos = 0
            for chunk in chunks:
                chunk_kv = chunk.kv_by_layer[layer_name]
                k_pre = chunk_kv[0]
                v = chunk_kv[1]
                token_count = k_pre.shape[0]

                k_pre_t = k_pre.transpose(0, 1).contiguous()
                k_rot_t = rotate_chunk_at_virtual_position(
                    k_pre_chunk=k_pre_t,
                    virtual_start=virtual_pos,
                    cos_table=self.cos_table,
                    sin_table=self.sin_table,
                )
                rotated_keys.append(k_rot_t.transpose(0, 1).contiguous())
                values.append(v)
                virtual_pos += token_count

            composed[layer_name] = torch.stack(
                [torch.cat(rotated_keys, dim=0), torch.cat(values, dim=0)],
                dim=0,
            )
        return composed


def _encode_token_chunk_pre_rope(hf_model: Any, token_ids: list[int]) -> dict[str, Any]:
    import torch
    from encode_memories_pre_rope import capture_pre_rope

    device = next(hf_model.parameters()).device
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad(), capture_pre_rope() as capture:
        outputs = hf_model(input_ids=input_ids, use_cache=True)

    post_rope_kv = outputs.past_key_values
    if hasattr(post_rope_kv, "layers"):
        post_pairs = [(layer.keys, layer.values) for layer in post_rope_kv.layers]
    elif hasattr(post_rope_kv, "key_cache"):
        post_pairs = list(zip(post_rope_kv.key_cache, post_rope_kv.value_cache))
    else:
        post_pairs = list(post_rope_kv)

    kv_by_layer = {}
    for layer_idx, (_, value_post) in enumerate(post_pairs):
        slot = capture.layers[layer_idx]
        k_pre = slot.k_pre.squeeze(0).transpose(0, 1).contiguous()
        value = value_post.squeeze(0).transpose(0, 1).contiguous()
        layer_name = f"model.layers.{layer_idx}.self_attn.attn"
        kv_by_layer[layer_name] = torch.stack([k_pre, value], dim=0)
    return kv_by_layer
