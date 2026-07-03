from __future__ import annotations

import logging
import time
from typing import Any

from ..data import ConversationSample, SessionChunk
from ..vector_types import SearchHit
from .context import build_session_context_encoding_plan
from .prompting import (
    MemoryOrder,
    memory_frame_prefix_token_ids,
    memory_frame_suffix_token_ids,
    ordered_memory_session_ids,
)
from .submodule import require_ai_memory_submodule
from .types import ComposedMemory, EncodedChunk

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


class SharedPreRopeEncoder:
    """One HF pre-RoPE encoder (model, tokenizer, RoPE tables) shared across samples."""

    def __init__(self, *, model: str, dtype: str, device: str, max_position: int) -> None:
        require_ai_memory_submodule()
        import torch
        from encode_memories_pre_rope import PreRoPEMemoryEncoder
        from rope_inject import extract_cos_sin_from_model

        if not torch.cuda.is_available():
            raise RuntimeError("KV injection requires a CUDA device.")

        self.model = model
        self.device = device
        self.max_position = max_position
        started = time.perf_counter()
        self._encoder = PreRoPEMemoryEncoder(
            model_name=model,
            dtype=torch_dtype(dtype),
            device=device,
        )
        self._encoder.load_model()
        self.tokenizer = self._encoder._tokenizer
        self.hf_model = self._encoder._model
        if self.tokenizer is None or self.hf_model is None:
            raise RuntimeError("Pre-RoPE encoder did not load a model/tokenizer.")

        cfg = self.hf_model.config
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        positions = torch.arange(max_position, device=device)
        self.cos_table, self.sin_table = extract_cos_sin_from_model(
            self.hf_model,
            positions,
            self.head_dim,
        )
        logger.info(
            "Loaded shared pre-RoPE encoder model=%s device=%s in %.1fs gpu_gb=%.1f",
            model,
            device,
            time.perf_counter() - started,
            torch.cuda.memory_allocated() / 1e9,
        )

    def release_weights(self) -> None:
        """Free encoder weights before vLLM starts; keep cos/sin tables for composition."""
        import gc
        import torch

        self._encoder = None
        self.hf_model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def close(self) -> None:
        self.release_weights()
        self.cos_table = None
        self.sin_table = None


class ChunkedRopeSampleComposer:
    """GPU-resident pre-RoPE chunk store and top-k composer for one sample."""

    def __init__(self, *, encoder: SharedPreRopeEncoder, context_window: int = 0) -> None:
        if context_window < 0:
            raise ValueError("context_window must be >= 0.")

        self.encoder = encoder
        self.device = encoder.device
        self.max_position = encoder.max_position
        self.context_window = context_window
        self.chunks: dict[str, EncodedChunk] = {}

        tokenizer, _ = self._require_encode_ready()
        prefix_token_ids = memory_frame_prefix_token_ids(tokenizer)
        self.prefix_chunk: EncodedChunk | None = self._encode_token_ids_chunk("__prefix__", prefix_token_ids)
        suffix_token_ids = memory_frame_suffix_token_ids(tokenizer)
        self.suffix_chunk: EncodedChunk | None = (
            self._encode_token_ids_chunk("__suffix__", suffix_token_ids)
            if suffix_token_ids
            else None
        )

    def encode_sample(self, sample: ConversationSample, session_ids: set[str] | None = None) -> None:
        sessions = sample.sessions
        if session_ids is not None:
            sessions = [
                session
                for session in sample.sessions
                if session.id in session_ids or session.session_id in session_ids
            ]
        logger.info(
            "Pre-RoPE encoding %d session memory chunks for sample_id=%s context_window=%d",
            len(sessions),
            sample.sample_id,
            self.context_window,
        )
        for session in sessions:
            self.chunks[session.id] = self._encode_session_chunk(sample, session)
        logger.info("Pre-RoPE encoded %d chunks for sample_id=%s", len(self.chunks), sample.sample_id)

    def compose(
        self,
        sample: ConversationSample,
        hits: list[SearchHit],
        *,
        memory_order: MemoryOrder = "retrieval",
    ) -> ComposedMemory:
        started = time.perf_counter()
        if self.encoder.cos_table is None or self.encoder.sin_table is None:
            raise RuntimeError("Pre-RoPE encoder was closed; cannot compose memory.")
        if self.prefix_chunk is None:
            raise RuntimeError("Composer was closed; cannot compose memory.")
        retrieval_session_ids, session_ids = ordered_memory_session_ids(sample, hits, memory_order=memory_order)

        selected = []
        missing = []
        for session_id in session_ids:
            chunk = self.chunks.get(session_id)
            if chunk is None:
                session = next(
                    (
                        candidate
                        for candidate in sample.sessions
                        if candidate.session_id == session_id or candidate.id == session_id
                    ),
                    None,
                )
                if session is not None:
                    chunk = self.chunks.get(session.id)
            if chunk is None:
                missing.append(session_id)
            else:
                selected.append(chunk)
        if missing:
            raise RuntimeError(
                "Retrieved memory chunks were not pre-encoded: " + ", ".join(missing[:5])
            )

        chunks = [self.prefix_chunk, *selected]
        if self.suffix_chunk is not None:
            chunks.append(self.suffix_chunk)
        kv_by_layer = _compose_encoded_chunks(
            chunks,
            device=self.device,
            max_position=self.max_position,
            cos_table=self.encoder.cos_table,
            sin_table=self.encoder.sin_table,
        )
        token_ids: list[int] = []
        for chunk in chunks:
            token_ids.extend(chunk.token_ids)
        selected_context_tokens = [chunk.context_prefix_tokens for chunk in selected]
        return ComposedMemory(
            kv_by_layer=kv_by_layer,
            token_ids=token_ids,
            num_tokens=len(token_ids),
            compose_time_ms=(time.perf_counter() - started) * 1000,
            retrieval_session_ids=retrieval_session_ids,
            selected_session_ids=session_ids,
            memory_order=memory_order,
            context_window=self.context_window,
            context_prefix_tokens_total=sum(selected_context_tokens),
            context_prefix_tokens_max=max(selected_context_tokens, default=0),
            context_prefix_truncated_tokens=sum(chunk.context_prefix_truncated_tokens for chunk in selected),
        )

    def cache_stats(self) -> dict[str, Any]:
        chunks = list(self.chunks.values())
        if self.prefix_chunk is not None:
            chunks.insert(0, self.prefix_chunk)
        if self.suffix_chunk is not None:
            chunks.append(self.suffix_chunk)

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
            "kv_precomputed_chunks": len(self.chunks),
            "kv_precomputed_chunks_with_prefix": len(chunks),
            "kv_precomputed_tokens": total_tokens,
            "kv_precomputed_layers": layer_count,
            "kv_precomputed_gpu_mb": total_bytes / (1024 * 1024),
            "kv_precomputed_devices": ",".join(sorted(devices)),
        }

    def close(self) -> None:
        """Free this sample's encoded chunks. The shared encoder is owned by the client."""
        import gc
        import torch

        self.chunks.clear()
        self.prefix_chunk = None
        self.suffix_chunk = None
        gc.collect()
        torch.cuda.empty_cache()

    def _require_encode_ready(self) -> tuple[Any, Any]:
        tokenizer = self.encoder.tokenizer
        hf_model = self.encoder.hf_model
        if tokenizer is None or hf_model is None:
            raise RuntimeError(
                "Pre-RoPE encoder weights were released. Encode all samples before starting vLLM."
            )
        return tokenizer, hf_model

    def _encode_session_chunk(self, sample: ConversationSample, session: SessionChunk) -> EncodedChunk:
        tokenizer, hf_model = self._require_encode_ready()
        plan = build_session_context_encoding_plan(
            tokenizer,
            sample,
            session,
            context_window=self.context_window,
            max_input_tokens=self.max_position,
        )
        return EncodedChunk(
            token_ids=plan.target_token_ids,
            kv_by_layer=_detach_kv_by_layer_to_device(
                _encode_token_chunk_pre_rope(
                    hf_model,
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

    def _encode_token_ids_chunk(self, chunk_id: str, token_ids: list[int]) -> EncodedChunk:
        _, hf_model = self._require_encode_ready()
        if not token_ids:
            raise RuntimeError(f"Memory chunk tokenized to zero tokens: {chunk_id}")
        return EncodedChunk(
            token_ids=list(token_ids),
            kv_by_layer=_detach_kv_by_layer_to_device(
                _encode_token_chunk_pre_rope(
                    hf_model,
                    token_ids,
                    slice_start=0,
                    slice_end=len(token_ids),
                ),
                self.device,
            ),
        )


def _compose_encoded_chunks(
    chunks: list[EncodedChunk],
    *,
    device: str,
    max_position: int,
    cos_table: Any,
    sin_table: Any,
) -> dict[str, Any]:
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

    first_kv = next(iter(chunks[0].kv_by_layer.values()))
    # [tokens, 1, head_dim] broadcasts over the kv-head axis of [tokens, heads, head_dim],
    # so K rotates in its stored layout without transpose round-trips.
    cos = cos_table[:total_tokens].unsqueeze(1).to(dtype=first_kv.dtype, device=device)
    sin = sin_table[:total_tokens].unsqueeze(1).to(dtype=first_kv.dtype, device=device)

    for layer_name in layer_names:
        layer_chunks = [
            _copy_layer_kv_to_device(chunk.kv_by_layer[layer_name], device)
            for chunk in chunks
        ]
        k_pre = torch.cat([chunk_kv[0] for chunk_kv in layer_chunks], dim=0)
        values = torch.cat([chunk_kv[1] for chunk_kv in layer_chunks], dim=0)
        k_rot = (k_pre * cos) + (_rotate_half(k_pre) * sin)

        composed[layer_name] = torch.stack(
            [k_rot, values],
            dim=0,
        )
    return composed


def _rotate_half(value: Any) -> Any:
    import torch

    half = value.shape[-1] // 2
    return torch.cat([-value[..., half:], value[..., :half]], dim=-1)


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

    if len(capture.layers) != len(post_pairs):
        raise RuntimeError(
            f"RoPE capture saw {len(capture.layers)} layers but model has {len(post_pairs)}."
        )

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
