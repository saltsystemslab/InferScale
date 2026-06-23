from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from pathlib import Path
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
    context_window: int = 0
    context_prefix_tokens: int = 0
    raw_context_prefix_tokens: int = 0
    context_prefix_truncated_tokens: int = 0
    encoding_input_tokens: int = 0


@dataclass(slots=True, frozen=True)
class ContextEncodingPlan:
    turn_id: str
    target_text: str
    target_token_ids: list[int]
    context_token_ids: list[int]
    input_token_ids: list[int]
    slice_start: int
    slice_end: int
    raw_context_prefix_tokens: int
    context_prefix_truncated_tokens: int


@dataclass(slots=True)
class ComposedMemory:
    kv_by_layer: dict[str, Any]
    token_ids: list[int]
    num_tokens: int
    compose_time_ms: float
    selected_turn_ids: list[str]
    context_window: int
    context_prefix_tokens_total: int
    context_prefix_tokens_max: int
    context_prefix_truncated_tokens: int


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
            turn_id=turn.id,
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
            text=plan.target_text,
            context_window=self.context_window,
            context_prefix_tokens=len(plan.context_token_ids),
            raw_context_prefix_tokens=plan.raw_context_prefix_tokens,
            context_prefix_truncated_tokens=plan.context_prefix_truncated_tokens,
            encoding_input_tokens=len(plan.input_token_ids),
        )

    def _encode_text_chunk(self, turn_id: str, text: str) -> EncodedChunk:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"Memory chunk tokenized to zero tokens: {turn_id}")
        return EncodedChunk(
            turn_id=turn_id,
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
            text=text,
            encoding_input_tokens=len(token_ids),
        )

    def _compose_chunks(self, chunks: list[EncodedChunk]) -> dict[str, Any]:
        return _compose_encoded_chunks(
            chunks,
            device=self.device,
            max_position=self.max_position,
            cos_table=self.cos_table,
            sin_table=self.sin_table,
        )


class CachedChunkedRopeSampleComposer:
    """CPU-resident chunk composer loaded from a precomputed sample KV cache."""

    def __init__(
        self,
        *,
        prefix_chunk: EncodedChunk,
        chunks: dict[str, EncodedChunk],
        cos_table: Any,
        sin_table: Any,
        device: str,
        max_position: int,
        context_window: int,
    ) -> None:
        self.prefix_chunk = prefix_chunk
        self.chunks = chunks
        self.cos_table = cos_table
        self.sin_table = sin_table
        self.device = device
        self.max_position = max_position
        self.context_window = context_window

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
                "Retrieved memory chunks were not found in the CPU KV cache: " + ", ".join(missing[:5])
            )

        chunks = [self.prefix_chunk, *selected]
        kv_by_layer = _compose_encoded_chunks(
            chunks,
            device=self.device,
            max_position=self.max_position,
            cos_table=_copy_layer_kv_to_device(self.cos_table, self.device),
            sin_table=_copy_layer_kv_to_device(self.sin_table, self.device),
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
            selected_turn_ids=turn_ids,
            context_window=self.context_window,
            context_prefix_tokens_total=sum(selected_context_tokens),
            context_prefix_tokens_max=max(selected_context_tokens, default=0),
            context_prefix_truncated_tokens=sum(chunk.context_prefix_truncated_tokens for chunk in selected),
        )

    def close(self) -> None:
        import gc

        for attr in ("prefix_chunk", "chunks", "cos_table", "sin_table"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()


def save_sample_kv_cache(path: Path, composer: ChunkedRopeSampleComposer) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "max_position": composer.max_position,
        "context_window": composer.context_window,
        "prefix_chunk": _chunk_to_cache_dict(composer.prefix_chunk, offload_to_cpu=True),
        "chunks": {
            turn_id: _chunk_to_cache_dict(chunk, offload_to_cpu=True)
            for turn_id, chunk in composer.chunks.items()
        },
        "cos_table": _offload_tensor_to_cpu(composer.cos_table),
        "sin_table": _offload_tensor_to_cpu(composer.sin_table),
    }
    torch.save(payload, path)


def load_sample_kv_cache(path: Path, *, device: str) -> CachedChunkedRopeSampleComposer:
    import torch

    payload = torch.load(path, map_location="cpu")
    if payload.get("version") != 1:
        raise RuntimeError(f"Unsupported sample KV cache version in {path}: {payload.get('version')!r}")
    return CachedChunkedRopeSampleComposer(
        prefix_chunk=_chunk_from_cache_dict(payload["prefix_chunk"]),
        chunks={
            str(turn_id): _chunk_from_cache_dict(chunk)
            for turn_id, chunk in payload["chunks"].items()
        },
        cos_table=payload["cos_table"],
        sin_table=payload["sin_table"],
        device=device,
        max_position=int(payload["max_position"]),
        context_window=int(payload["context_window"]),
    )


def _compose_encoded_chunks(
    chunks: list[EncodedChunk],
    *,
    device: str,
    max_position: int,
    cos_table: Any,
    sin_table: Any,
) -> dict[str, Any]:
    from rope_inject import rotate_chunk_at_virtual_position
    import torch

    layer_names = list(chunks[0].kv_by_layer.keys())
    composed: dict[str, Any] = {}
    total_tokens = sum(len(chunk.token_ids) for chunk in chunks)
    if total_tokens > max_position:
        raise RuntimeError(
            f"Composed memory has {total_tokens} tokens, exceeding kv_max_position={max_position}."
        )

    for layer_name in layer_names:
        rotated_keys = []
        values = []
        virtual_pos = 0
        for chunk in chunks:
            chunk_kv = _copy_layer_kv_to_device(chunk.kv_by_layer[layer_name], device)
            k_pre = chunk_kv[0]
            v = chunk_kv[1]
            token_count = k_pre.shape[0]

            k_pre_t = k_pre.transpose(0, 1).contiguous()
            k_rot_t = rotate_chunk_at_virtual_position(
                k_pre_chunk=k_pre_t,
                virtual_start=virtual_pos,
                cos_table=cos_table,
                sin_table=sin_table,
            )
            rotated_keys.append(k_rot_t.transpose(0, 1).contiguous())
            values.append(v)
            virtual_pos += token_count

        composed[layer_name] = torch.stack(
            [torch.cat(rotated_keys, dim=0), torch.cat(values, dim=0)],
            dim=0,
        )
    return composed


def _chunk_to_cache_dict(chunk: EncodedChunk, *, offload_to_cpu: bool = False) -> dict[str, Any]:
    kv_by_layer = (
        _offload_kv_by_layer_to_cpu(chunk.kv_by_layer)
        if offload_to_cpu
        else chunk.kv_by_layer
    )
    return {
        "turn_id": chunk.turn_id,
        "token_ids": list(chunk.token_ids),
        "kv_by_layer": kv_by_layer,
        "text": chunk.text,
        "context_window": chunk.context_window,
        "context_prefix_tokens": chunk.context_prefix_tokens,
        "raw_context_prefix_tokens": chunk.raw_context_prefix_tokens,
        "context_prefix_truncated_tokens": chunk.context_prefix_truncated_tokens,
        "encoding_input_tokens": chunk.encoding_input_tokens,
    }


def _chunk_from_cache_dict(data: dict[str, Any]) -> EncodedChunk:
    return EncodedChunk(
        turn_id=str(data["turn_id"]),
        token_ids=list(data["token_ids"]),
        kv_by_layer=dict(data["kv_by_layer"]),
        text=str(data.get("text", "")),
        context_window=int(data.get("context_window", 0)),
        context_prefix_tokens=int(data.get("context_prefix_tokens", 0)),
        raw_context_prefix_tokens=int(data.get("raw_context_prefix_tokens", 0)),
        context_prefix_truncated_tokens=int(data.get("context_prefix_truncated_tokens", 0)),
        encoding_input_tokens=int(data.get("encoding_input_tokens", 0)),
    )


def build_turn_context_encoding_plan(
    tokenizer: Any,
    sample: ConversationSample,
    turn: Turn,
    *,
    context_window: int,
    max_input_tokens: int,
) -> ContextEncodingPlan:
    if context_window < 0:
        raise ValueError("context_window must be >= 0.")
    if max_input_tokens < 1:
        raise ValueError("max_input_tokens must be >= 1.")

    target_text = format_memory_turn(turn)
    target_token_ids = _encode_text_no_special(tokenizer, target_text)
    if not target_token_ids:
        raise RuntimeError(f"Memory chunk tokenized to zero tokens: {turn.id}")
    if len(target_token_ids) > max_input_tokens:
        raise RuntimeError(
            f"Memory turn {turn.id} has {len(target_token_ids)} tokens, "
            f"exceeding kv_max_position={max_input_tokens} even without context."
        )

    context_token_ids: list[int] = []
    for context_turn in previous_session_context_turns(sample, turn, context_window):
        context_token_ids.extend(_encode_text_no_special(tokenizer, format_memory_turn(context_turn)))

    raw_context_prefix_tokens = len(context_token_ids)
    overflow = len(context_token_ids) + len(target_token_ids) - max_input_tokens
    context_prefix_truncated_tokens = max(0, overflow)
    if context_prefix_truncated_tokens:
        context_token_ids = context_token_ids[context_prefix_truncated_tokens:]

    input_token_ids = context_token_ids + target_token_ids
    slice_start = len(context_token_ids)
    slice_end = len(input_token_ids)
    return ContextEncodingPlan(
        turn_id=turn.id,
        target_text=target_text,
        target_token_ids=target_token_ids,
        context_token_ids=context_token_ids,
        input_token_ids=input_token_ids,
        slice_start=slice_start,
        slice_end=slice_end,
        raw_context_prefix_tokens=raw_context_prefix_tokens,
        context_prefix_truncated_tokens=context_prefix_truncated_tokens,
    )


def previous_session_context_turns(sample: ConversationSample, turn: Turn, context_window: int) -> list[Turn]:
    if context_window <= 0:
        return []
    first_session_index = turn.session_index - context_window
    return [
        candidate
        for candidate in sample.turns
        if first_session_index <= candidate.session_index < turn.session_index
    ]


def format_memory_turn(turn: Turn) -> str:
    return format_turn_for_memory(turn).strip() + "\n"


def _encode_text_no_special(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Tokenizer has no encode method.")
    return list(encode(text, add_special_tokens=False))


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


def _offload_kv_by_layer_to_cpu(kv_by_layer: dict[str, Any]) -> dict[str, Any]:
    return {
        layer_name: _offload_tensor_to_cpu(tensor)
        for layer_name, tensor in kv_by_layer.items()
    }


def _detach_kv_by_layer_to_device(kv_by_layer: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        layer_name: _detach_tensor_to_device(tensor, device)
        for layer_name, tensor in kv_by_layer.items()
    }


def _detach_tensor_to_device(tensor: Any, device: str) -> Any:
    detached = tensor.detach().to(device=device, non_blocking=True)
    return detached.contiguous()


def _offload_tensor_to_cpu(tensor: Any) -> Any:
    import torch

    detached = tensor.detach()
    try:
        host_tensor = torch.empty(
            detached.shape,
            dtype=detached.dtype,
            pin_memory=True,
        )
    except RuntimeError:
        return detached.to(device="cpu").contiguous()
    host_tensor.copy_(detached, non_blocking=False)
    return host_tensor


def _copy_layer_kv_to_device(layer_kv: Any, device: str) -> Any:
    return layer_kv.to(device=device, non_blocking=True)
