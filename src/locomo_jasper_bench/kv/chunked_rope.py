from __future__ import annotations

import time
import logging
from typing import Any

from ..data import ConversationSample, Turn
from ..vector_types import SearchHit
from .context import build_turn_context_encoding_plan, format_memory_turn, previous_session_context_turns
from .prompting import MEMORY_PREFIX_TEXT, selected_turn_ids
from .submodule import require_ai_memory_submodule
from .tokenization import encode_text_no_special
from .types import ComposedMemory, ContextEncodingPlan, EncodedChunk

logger = logging.getLogger(__name__)


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
        context_window: int = 0,
    ) -> None:
        require_ai_memory_submodule()
        import torch
        from encode_memories_pre_rope import PreRoPEMemoryEncoder
        from rope_inject import extract_cos_sin_from_model

        if not torch.cuda.is_available():
            raise RuntimeError("KV injection requires a CUDA device.")
        if context_window < 0:
            raise ValueError("context_window must be >= 0.")

        self.model = model
        self.device = device
        self.max_position = max_position
        self.context_window = context_window
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
        logger.info(
            "Pre-RoPE encoding %d memory chunks for sample_id=%s context_window=%d",
            len(turns),
            sample.sample_id,
            self.context_window,
        )
        for turn in turns:
            self.chunks[turn.id] = self._encode_turn_chunk(sample, turn)
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
        selected_context_tokens = [chunk.context_prefix_tokens for chunk in selected]
        return ComposedMemory(
            kv_by_layer=kv_by_layer,
            token_ids=token_ids,
            num_tokens=len(token_ids),
            compose_time_ms=(time.perf_counter() - started) * 1000,
            selected_turn_ids=turn_ids,
            context_window=self.context_window,
            context_prefix_tokens_total=sum(selected_context_tokens),
            context_prefix_tokens_max=max(selected_context_tokens, default=0),
            context_prefix_truncated_tokens=sum(chunk.context_prefix_truncated_tokens for chunk in selected),
        )

    def cache_stats(self) -> dict[str, Any]:
        chunks = []
        prefix_chunk = getattr(self, "prefix_chunk", None)
        if prefix_chunk is not None:
            chunks.append(prefix_chunk)
        chunks.extend(getattr(self, "chunks", {}).values())

        total_bytes = 0
        total_tokens = 0
        layer_count = 0
        devices: set[str] = set()
        for chunk in chunks:
            total_tokens += len(chunk.token_ids)
            layer_count = max(layer_count, len(chunk.kv_by_layer))
            for tensor in chunk.kv_by_layer.values():
                total_bytes += int(getattr(tensor, "nbytes", 0) or 0)
                device = getattr(tensor, "device", None)
                if device is not None:
                    devices.add(str(device))

        return {
            "kv_chunk_cache_residency": "gpu",
            "kv_precomputed_chunks": max(0, len(chunks) - 1),
            "kv_precomputed_chunks_with_prefix": len(chunks),
            "kv_precomputed_tokens": total_tokens,
            "kv_precomputed_layers": layer_count,
            "kv_precomputed_gpu_mb": total_bytes / (1024 * 1024),
            "kv_precomputed_devices": ",".join(sorted(devices)),
        }

    def release_encoder(self) -> None:
        """Unload HF encoder weights while keeping encoded GPU chunks resident."""
        import gc
        import torch

        for attr in ("encoder", "hf_model", "tokenizer"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()

    def close(self) -> None:
        import gc
        import torch

        self.release_encoder()
        for attr in ("encoder", "hf_model", "tokenizer", "cos_table", "sin_table", "chunks", "prefix_chunk"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()

    def _encode_turn_chunk(self, sample: ConversationSample, turn: Turn) -> EncodedChunk:
        plan = build_turn_context_encoding_plan(
            self.tokenizer,
            sample,
            turn,
            context_window=self.context_window,
            max_input_tokens=self.max_position,
        )
        return EncodedChunk(
            token_ids=plan.target_token_ids,
            kv_by_layer=_detach_kv_by_layer_to_device(
                _encode_token_chunk_pre_rope(
                    self.hf_model,
                    plan.input_token_ids,
                    slice_start=plan.slice_start,
                    slice_end=plan.slice_end,
                ),
                self.device,
            ),
            context_window=self.context_window,
            context_prefix_tokens=len(plan.context_token_ids),
            raw_context_prefix_tokens=plan.raw_context_prefix_tokens,
            context_prefix_truncated_tokens=plan.context_prefix_truncated_tokens,
        )

    def _encode_text_chunk(self, turn_id: str, text: str) -> EncodedChunk:
        token_ids = encode_text_no_special(self.tokenizer, text)
        if not token_ids:
            raise RuntimeError(f"Memory chunk tokenized to zero tokens: {turn_id}")
        return EncodedChunk(
            token_ids=token_ids,
            kv_by_layer=_detach_kv_by_layer_to_device(
                _encode_token_chunk_pre_rope(
                    self.hf_model,
                    token_ids,
                    slice_start=0,
                    slice_end=len(token_ids),
                ),
                self.device,
            ),
        )

    def _compose_chunks(self, chunks: list[EncodedChunk]) -> dict[str, Any]:
        return _compose_encoded_chunks(
            chunks,
            device=self.device,
            max_position=self.max_position,
            cos_table=self.cos_table,
            sin_table=self.sin_table,
        )


def _compose_encoded_chunks(
    chunks: list[EncodedChunk],
    *,
    device: str,
    max_position: int,
    cos_table: Any,
    sin_table: Any,
) -> dict[str, Any]:
    from rope_inject import rotate_pre_rope_k
    import torch

    layer_names = list(chunks[0].kv_by_layer.keys())
    composed: dict[str, Any] = {}
    total_tokens = sum(len(chunk.token_ids) for chunk in chunks)
    if total_tokens > max_position:
        raise RuntimeError(
            f"Composed memory has {total_tokens} tokens, exceeding kv_max_position={max_position}."
        )
    if total_tokens > cos_table.shape[0] or total_tokens > sin_table.shape[0]:
        raise ValueError(
            f"Composed memory has {total_tokens} tokens, exceeding precomputed RoPE table size "
            f"cos={cos_table.shape[0]} sin={sin_table.shape[0]}."
        )

    for layer_name in layer_names:
        layer_chunks = [
            _copy_layer_kv_to_device(chunk.kv_by_layer[layer_name], device)
            for chunk in chunks
        ]
        k_pre = torch.cat([chunk_kv[0] for chunk_kv in layer_chunks], dim=0)
        values = torch.cat([chunk_kv[1] for chunk_kv in layer_chunks], dim=0)
        k_rot = rotate_pre_rope_k(
            k_pre.transpose(0, 1).contiguous(),
            cos_table[:total_tokens],
            sin_table[:total_tokens],
        ).transpose(0, 1).contiguous()

        composed[layer_name] = torch.stack(
            [k_rot, values],
            dim=0,
        )
    return composed


def _encode_token_chunk_pre_rope(
    hf_model: Any,
    token_ids: list[int],
    *,
    slice_start: int,
    slice_end: int,
) -> dict[str, Any]:
    import torch
    from encode_memories_pre_rope import capture_pre_rope

    if slice_start < 0 or slice_end < slice_start or slice_end > len(token_ids):
        raise ValueError("Invalid KV slice bounds for pre-RoPE encoding.")

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
        k_pre = (
            slot.k_pre.squeeze(0)[:, slice_start:slice_end, :]
            .transpose(0, 1)
            .contiguous()
        )
        value = (
            value_post.squeeze(0)[:, slice_start:slice_end, :]
            .transpose(0, 1)
            .contiguous()
        )
        layer_name = f"model.layers.{layer_idx}.self_attn.attn"
        kv_by_layer[layer_name] = torch.stack([k_pre, value], dim=0)
    return kv_by_layer


def _detach_kv_by_layer_to_device(kv_by_layer: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        layer_name: _detach_tensor_to_device(tensor, device)
        for layer_name, tensor in kv_by_layer.items()
    }


def _detach_tensor_to_device(tensor: Any, device: str) -> Any:
    detached = tensor.detach().to(device=device, non_blocking=True)
    return detached.contiguous()


def _copy_layer_kv_to_device(layer_kv: Any, device: str) -> Any:
    return layer_kv.to(device=device, non_blocking=True)
