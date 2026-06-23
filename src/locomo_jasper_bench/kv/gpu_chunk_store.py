from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any, Iterable

from ..vector_types import SearchHit
from .chunked_rope import EncodedChunk
from .prompting import selected_turn_ids
from .submodule import require_ai_memory_submodule


@dataclass(slots=True, frozen=True)
class GpuChunkSpan:
    chunk: EncodedChunk
    chunk_start: int
    length: int
    dst_start: int


@dataclass(slots=True)
class GpuChunkPlan:
    plan_id: str
    sample_id: str
    token_ids: list[int]
    chunks: tuple[EncodedChunk, ...]
    cos_table: Any
    sin_table: Any
    selected_turn_ids: list[str]
    context_window: int
    context_prefix_tokens_total: int
    context_prefix_tokens_max: int
    context_prefix_truncated_tokens: int
    plan_time_ms: float

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def iter_spans(self, max_tokens: int | None = None) -> Iterable[GpuChunkSpan]:
        remaining = self.num_tokens if max_tokens is None else min(max_tokens, self.num_tokens)
        dst_start = 0
        for chunk in self.chunks:
            if remaining <= 0:
                break
            length = min(len(chunk.token_ids), remaining)
            if length > 0:
                yield GpuChunkSpan(
                    chunk=chunk,
                    chunk_start=0,
                    length=length,
                    dst_start=dst_start,
                )
            remaining -= length
            dst_start += length

    def layer_span_kv(self, layer_name: str, span: GpuChunkSpan) -> Any:
        require_ai_memory_submodule()
        import torch
        from rope_inject import rotate_chunk_at_virtual_position

        layer_kv = span.chunk.kv_by_layer.get(layer_name)
        if layer_kv is None:
            raise KeyError(f"Layer {layer_name} not found in chunk {span.chunk.turn_id}.")
        if layer_kv.ndim < 4 or layer_kv.shape[0] != 2:
            raise ValueError(
                f"Expected chunk KV for {span.chunk.turn_id}/{layer_name} to have shape [2, tokens, ...], "
                f"got {tuple(layer_kv.shape)}"
            )

        end = span.chunk_start + span.length
        k_pre = layer_kv[0, span.chunk_start:end]
        value = layer_kv[1, span.chunk_start:end]
        k_pre_t = k_pre.transpose(0, 1).contiguous()
        k_rot_t = rotate_chunk_at_virtual_position(
            k_pre_chunk=k_pre_t,
            virtual_start=span.dst_start,
            cos_table=self.cos_table,
            sin_table=self.sin_table,
        )
        return torch.stack(
            [k_rot_t.transpose(0, 1).contiguous(), value.contiguous()],
            dim=0,
        )


class GpuSampleChunkStore:
    """Strict GPU-resident pre-RoPE chunks for one LoCoMo sample."""

    def __init__(
        self,
        *,
        sample_id: str,
        prefix_chunk: EncodedChunk,
        chunks: dict[str, EncodedChunk],
        cos_table: Any,
        sin_table: Any,
        device: str,
        max_position: int,
        context_window: int,
    ) -> None:
        self.sample_id = sample_id
        self.prefix_chunk = prefix_chunk
        self.chunks = chunks
        self.cos_table = cos_table
        self.sin_table = sin_table
        self.device = device
        self.max_position = max_position
        self.context_window = context_window

    def build_plan(self, *, plan_id: str, hits: list[SearchHit]) -> GpuChunkPlan:
        started = time.perf_counter()
        turn_ids = selected_turn_ids(hits)
        if not turn_ids:
            raise RuntimeError("Cannot build GPU chunk plan because retrieval returned no turn ids.")

        selected: list[EncodedChunk] = []
        missing: list[str] = []
        for turn_id in turn_ids:
            chunk = self.chunks.get(turn_id)
            if chunk is None:
                missing.append(turn_id)
            else:
                selected.append(chunk)
        if missing:
            raise RuntimeError(
                "Retrieved memory chunks were not found in the GPU chunk store: " + ", ".join(missing[:5])
            )

        ordered_chunks = (self.prefix_chunk, *selected)
        token_ids: list[int] = []
        for chunk in ordered_chunks:
            token_ids.extend(chunk.token_ids)
        if len(token_ids) > self.max_position:
            raise RuntimeError(
                f"Composed memory has {len(token_ids)} tokens, exceeding kv_max_position={self.max_position}."
            )

        selected_context_tokens = [chunk.context_prefix_tokens for chunk in selected]
        return GpuChunkPlan(
            plan_id=plan_id,
            sample_id=self.sample_id,
            token_ids=token_ids,
            chunks=ordered_chunks,
            cos_table=self.cos_table,
            sin_table=self.sin_table,
            selected_turn_ids=turn_ids,
            context_window=self.context_window,
            context_prefix_tokens_total=sum(selected_context_tokens),
            context_prefix_tokens_max=max(selected_context_tokens, default=0),
            context_prefix_truncated_tokens=sum(chunk.context_prefix_truncated_tokens for chunk in selected),
            plan_time_ms=(time.perf_counter() - started) * 1000,
        )

    def get_stats(self) -> dict[str, float | int]:
        total_bytes = _tensor_nbytes(self.cos_table) + _tensor_nbytes(self.sin_table)
        total_tokens = len(self.prefix_chunk.token_ids)
        total_bytes += _chunk_nbytes(self.prefix_chunk)
        for chunk in self.chunks.values():
            total_tokens += len(chunk.token_ids)
            total_bytes += _chunk_nbytes(chunk)
        return {
            "sample_chunks": len(self.chunks),
            "sample_chunk_tokens": total_tokens,
            "sample_gpu_mb": total_bytes / (1024 * 1024),
        }

    def close(self) -> None:
        for attr in ("prefix_chunk", "chunks", "cos_table", "sin_table"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _chunk_nbytes(chunk: EncodedChunk) -> int:
    return sum(_tensor_nbytes(tensor) for tensor in chunk.kv_by_layer.values())


def _tensor_nbytes(tensor: Any) -> int:
    value = getattr(tensor, "nbytes", None)
    if value is not None:
        return int(value)
    element_size = getattr(tensor, "element_size", None)
    nelement = getattr(tensor, "nelement", None)
    if callable(element_size) and callable(nelement):
        return int(element_size() * nelement())
    return 0
