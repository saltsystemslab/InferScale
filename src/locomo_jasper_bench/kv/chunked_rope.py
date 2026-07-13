from __future__ import annotations

import sys
import time
import logging
from collections.abc import Sequence
from typing import Any

from ..data import ConversationSample
from ..vector_types import SearchHit
from .context import (
    build_fact_context_encoding_plan,
    build_memory_fact_plan,
    reverse_ranked_memory_facts,
    unique_memory_facts,
)
from .gpu_memory_store import bytes_to_mb
from .prompting import extract_memory_scaffold_token_ids, format_memory_fact
from .rope import extract_cos_sin_from_model
from .tokenization import encode_text_no_special
from .types import ComposedMemory, EncodedChunk, FactContextEncodingPlan, MemoryFact

logger = logging.getLogger(__name__)


def torch_dtype(dtype_name: str) -> Any:
    import torch

    normalized = dtype_name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported KV dtype: {dtype_name!r}")


class ChunkedRopeEncoder:
    """Shared HF pre-RoPE encoder used while precomputing sample chunks."""

    def __init__(
        self,
        *,
        model: str,
        dtype: str,
        device: str,
        max_position: int,
    ) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("KV injection requires a CUDA device.")

        self.model = model
        self.device = device
        self.max_position = max_position
        self.tokenizer, self.hf_model = _load_hf_model_and_tokenizer(
            model=model,
            dtype=torch_dtype(dtype),
            device=device,
        )

        cfg = self.hf_model.config
        self.head_dim = getattr(cfg, "head_dim", None)
        if self.head_dim is None:
            self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        positions = torch.arange(max_position, device=device)
        self.cos_table, self.sin_table = extract_cos_sin_from_model(
            self.hf_model,
            positions,
            self.head_dim,
        )

    def encode_token_ids_chunk(self, chunk_id: str, token_ids: list[int]) -> EncodedChunk:
        if self.hf_model is None:
            raise RuntimeError("Cannot encode KV chunk because the HF encoder has been released.")
        if not token_ids:
            raise RuntimeError(f"Memory chunk tokenized to zero tokens: {chunk_id}")
        if len(token_ids) > self.max_position:
            raise RuntimeError(
                f"Memory chunk {chunk_id} has {len(token_ids)} tokens, "
                f"exceeding kv_max_position={self.max_position}."
            )
        return EncodedChunk(
            token_ids=list(token_ids),
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

    def compose_chunks(self, chunks: list[EncodedChunk]) -> dict[str, Any]:
        """Compose pre-RoPE chunks into position-correct GPU KV tensors."""
        if not chunks:
            raise ValueError("At least one encoded chunk is required.")
        return _compose_encoded_chunks(
            chunks,
            device=self.device,
            max_position=self.max_position,
            cos_table=self.cos_table,
            sin_table=self.sin_table,
        )

    def encode_fact_chunk(self, plan: FactContextEncodingPlan) -> EncodedChunk:
        if self.hf_model is None:
            raise RuntimeError("Cannot encode KV chunk because the HF encoder has been released.")
        return EncodedChunk(
            token_ids=list(plan.target_token_ids),
            kv_by_layer=_detach_kv_by_layer_to_device(
                _encode_token_chunk_pre_rope(
                    self.hf_model,
                    plan.input_token_ids,
                    slice_start=plan.slice_start,
                    slice_end=plan.slice_end,
                ),
                self.device,
            ),
            context_turn_ids=plan.context_turn_ids,
            context_prefix_tokens=len(plan.context_token_ids),
            raw_context_prefix_tokens=plan.raw_context_tokens,
            context_prefix_truncated_tokens=plan.context_truncated_tokens,
        )

    def release_model(self) -> None:
        """Unload HF encoder weights while keeping RoPE tables available."""
        import gc
        import torch

        for attr in ("hf_model", "tokenizer"):
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

        self.release_model()
        for attr in ("cos_table", "sin_table"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()


class ChunkedRopeSampleComposer:
    """GPU-resident pre-RoPE chunk encoder and top-k composer for one sample."""

    def __init__(
        self,
        *,
        encoder: ChunkedRopeEncoder,
        context_window: int = 0,
        block_size: int = 16,
    ) -> None:
        if context_window < 0:
            raise ValueError("context_window must be >= 0.")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")

        self.encoder = encoder
        self.model = encoder.model
        self.device = encoder.device
        self.max_position = encoder.max_position
        self.context_window = context_window
        self.block_size = block_size
        self.chunks: dict[str, EncodedChunk] = {}
        self._fact_token_ids: dict[str, list[int]] = {}
        self._turn_token_ids: dict[str, list[int]] = {}
        self._facts: list[MemoryFact] = []
        self._facts_by_id: dict[str, MemoryFact] = {}
        self._sample: ConversationSample | None = None
        self.header_chunk: EncodedChunk | None = None
        self.memory_list_header_chunk: EncodedChunk | None = None
        self.empty_memory_chunk: EncodedChunk | None = None
        self.footer_chunk: EncodedChunk | None = None
        if encoder.tokenizer is None:
            raise RuntimeError("Cannot create sample composer because the HF encoder has been released.")

    def encode_sample(
        self,
        sample: ConversationSample,
        facts: Sequence[SearchHit],
    ) -> None:
        if self.encoder.tokenizer is None:
            raise RuntimeError("Cannot encode sample because the HF encoder has been released.")
        self._sample = sample
        self._facts = unique_memory_facts(facts)
        self._facts_by_id = {fact.memory_id: fact for fact in self._facts}
        scaffold = extract_memory_scaffold_token_ids(
            self.encoder.tokenizer,
            sample,
            block_size=self.block_size,
        )
        self.header_chunk = self.encoder.encode_token_ids_chunk(
            "__header__",
            scaffold.header_token_ids,
        )
        if scaffold.memory_list_header_token_ids:
            self.memory_list_header_chunk = self.encoder.encode_token_ids_chunk(
                "__memory_list_header__",
                scaffold.memory_list_header_token_ids,
            )
        else:
            # The scaffold renders retrieved facts without a list heading, so
            # there is no chunk to encode; compose() treats None as zero tokens.
            self.memory_list_header_chunk = None
        self.empty_memory_chunk = self.encoder.encode_token_ids_chunk(
            "__empty_memory__",
            scaffold.empty_memory_token_ids,
        )
        self.footer_chunk = self.encoder.encode_token_ids_chunk(
            "__footer__",
            scaffold.footer_token_ids,
        )
        self._fact_token_ids = {
            fact.memory_id: encode_text_no_special(
                self.encoder.tokenizer,
                format_memory_fact(fact),
            )
            for fact in self._facts
        }
        logger.info(
            "Pre-RoPE encoding %d unique Mem0 fact chunks for sample_id=%s context_window=%d",
            len(self._facts),
            sample.sample_id,
            self.context_window,
        )
        for fact in self._facts:
            self.chunks[fact.memory_id] = self._encode_fact_chunk(fact)
        logger.info("Pre-RoPE encoded %d chunks for sample_id=%s", len(self.chunks), sample.sample_id)

    def compose(
        self,
        hits: list[SearchHit],
        *,
        memory_token_budget: int | None = None,
    ) -> ComposedMemory:
        started = time.perf_counter()
        if self._sample is None:
            raise RuntimeError("Cannot compose KV memory before encoding a sample.")
        if (
            self.header_chunk is None
            or self.empty_memory_chunk is None
            or self.footer_chunk is None
        ):
            raise RuntimeError("Cannot compose KV memory before encoding its prompt scaffold.")
        retrieved_facts = unique_memory_facts(hits)
        selected_facts = reverse_ranked_memory_facts(hits)
        for fact in selected_facts:
            cached = self._facts_by_id.get(fact.memory_id)
            if cached is None:
                raise RuntimeError(
                    f"Retrieved Mem0 fact was not present in the precompute catalog: {fact.memory_id}."
                )
            if cached != fact:
                raise RuntimeError(
                    f"Retrieved Mem0 fact changed after precompute: {fact.memory_id}."
                )

        effective_memory_token_budget = min(
            self.max_position,
            self.max_position if memory_token_budget is None else memory_token_budget,
        )
        if selected_facts:
            memory_heading_chunks = (
                [self.memory_list_header_chunk]
                if self.memory_list_header_chunk is not None
                else []
            )
        else:
            memory_heading_chunks = [self.empty_memory_chunk]
        memory_heading_tokens = sum(
            len(chunk.token_ids) for chunk in memory_heading_chunks
        )
        scaffold_token_count = (
            len(self.header_chunk.token_ids)
            + memory_heading_tokens
            + len(self.footer_chunk.token_ids)
        )
        fact_plan = build_memory_fact_plan(
            selected_facts,
            retrieved_facts=retrieved_facts,
            context_window=self.context_window,
            memory_token_budget=effective_memory_token_budget,
            scaffold_token_count=scaffold_token_count,
            fact_token_ids=self._fact_token_ids,
            encoded_chunks=self.chunks,
        )
        if fact_plan.memory_tokens > effective_memory_token_budget:
            raise AssertionError("Planned KV memory exceeds its token budget.")

        fact_tokens_end = (
            len(self.header_chunk.token_ids)
            + memory_heading_tokens
            + fact_plan.fact_tokens
        )
        loaded_memory_tokens = (
            fact_plan.memory_tokens // self.block_size
        ) * self.block_size
        recomputed_memory_tail_tokens = fact_plan.memory_tokens - loaded_memory_tokens
        if loaded_memory_tokens < fact_tokens_end:
            raise RuntimeError(
                "The block-aligned KV prefix would leave retrieved fact tokens in the "
                "recomputed tail: "
                f"loaded={loaded_memory_tokens} facts_end={fact_tokens_end} "
                f"memory={fact_plan.memory_tokens} block_size={self.block_size}."
            )

        selected: list[EncodedChunk] = []
        for fact in selected_facts:
            selected.append(self.chunks[fact.memory_id])

        chunks = [self.header_chunk, *memory_heading_chunks, *selected]
        chunks.append(self.footer_chunk)
        kv_by_layer = self._compose_chunks(chunks)
        token_ids: list[int] = []
        for chunk in chunks:
            token_ids.extend(chunk.token_ids)
        if len(token_ids) != fact_plan.memory_tokens:
            raise AssertionError(
                "Planned memory token count does not match the composed KV memory."
            )
        return ComposedMemory(
            kv_by_layer=kv_by_layer,
            token_ids=token_ids,
            num_tokens=len(token_ids),
            compose_time_ms=(time.perf_counter() - started) * 1000,
            fact_plan=fact_plan,
            loaded_memory_tokens=loaded_memory_tokens,
            recomputed_memory_tail_tokens=recomputed_memory_tail_tokens,
            fact_tokens_end=fact_tokens_end,
        )

    def cache_stats(self) -> dict[str, Any]:
        scaffold_chunks = []
        header_chunk = getattr(self, "header_chunk", None)
        footer_chunk = getattr(self, "footer_chunk", None)
        memory_list_header_chunk = getattr(self, "memory_list_header_chunk", None)
        empty_memory_chunk = getattr(self, "empty_memory_chunk", None)
        if header_chunk is not None:
            scaffold_chunks.append(header_chunk)
        if memory_list_header_chunk is not None:
            scaffold_chunks.append(memory_list_header_chunk)
        if empty_memory_chunk is not None:
            scaffold_chunks.append(empty_memory_chunk)
        if footer_chunk is not None:
            scaffold_chunks.append(footer_chunk)
        fact_chunks_by_id = getattr(self, "chunks", {}) or {}
        fact_chunks = list(fact_chunks_by_id.values())
        chunks = list(scaffold_chunks)
        chunks.extend(fact_chunks)

        total_bytes = 0
        total_tokens = 0
        layer_count = 0
        devices: set[str] = set()
        for chunk in chunks:
            total_tokens += len(chunk.token_ids)
            layer_count = max(layer_count, len(chunk.kv_by_layer))
            for tensor in chunk.kv_by_layer.values():
                total_bytes += _tensor_nbytes(tensor)
                device = getattr(tensor, "device", None)
                if device is not None:
                    devices.add(str(device))
        prefix_tensor_bytes = sum(_chunk_tensor_bytes(chunk) for chunk in scaffold_chunks)
        fact_chunk_tensor_bytes = sum(_chunk_tensor_bytes(chunk) for chunk in fact_chunks)
        chunk_map_cpu_bytes = _chunk_map_cpu_bytes(fact_chunks_by_id)

        return {
            "kv_chunk_cache_residency": "gpu",
            "kv_precomputed_chunks": max(0, len(chunks) - len(scaffold_chunks)),
            "kv_precomputed_chunks_with_prefix": len(chunks),
            "kv_precomputed_tokens": total_tokens,
            "kv_precomputed_layers": layer_count,
            "kv_precomputed_gpu_mb": total_bytes / (1024 * 1024),
            "kv_precomputed_devices": ",".join(sorted(devices)),
            "llama_kv_chunk_count": len(fact_chunks),
            "llama_kv_chunk_map_cpu_bytes": chunk_map_cpu_bytes,
            "llama_kv_chunk_map_cpu_mb": bytes_to_mb(chunk_map_cpu_bytes),
            "llama_kv_chunk_tensor_gpu_bytes": fact_chunk_tensor_bytes,
            "llama_kv_chunk_tensor_gpu_mb": bytes_to_mb(fact_chunk_tensor_bytes),
            "llama_kv_prefix_tensor_gpu_bytes": prefix_tensor_bytes,
            "llama_kv_prefix_tensor_gpu_mb": bytes_to_mb(prefix_tensor_bytes),
            "llama_kv_total_tensor_gpu_bytes": total_bytes,
            "llama_kv_total_tensor_gpu_mb": bytes_to_mb(total_bytes),
        }

    def close(self) -> None:
        import gc
        import torch

        for attr in (
            "encoder",
            "chunks",
            "header_chunk",
            "memory_list_header_chunk",
            "empty_memory_chunk",
            "footer_chunk",
            "_fact_token_ids",
            "_turn_token_ids",
            "_facts",
            "_facts_by_id",
            "_sample",
        ):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()

    def _encode_fact_chunk(self, fact: MemoryFact) -> EncodedChunk:
        if self._sample is None:
            raise RuntimeError("Cannot encode a fact chunk before encode_sample().")
        plan = build_fact_context_encoding_plan(
            fact,
            self._sample,
            tokenizer=self.encoder.tokenizer,
            context_window=self.context_window,
            max_input_tokens=self.max_position,
            fact_token_ids=self._fact_token_ids,
            turn_token_ids=self._turn_token_ids,
        )
        return self.encoder.encode_fact_chunk(plan)

    def _compose_chunks(self, chunks: list[EncodedChunk]) -> dict[str, Any]:
        return self.encoder.compose_chunks(chunks)


def _compose_encoded_chunks(
    chunks: list[EncodedChunk],
    *,
    device: str,
    max_position: int,
    cos_table: Any,
    sin_table: Any,
) -> dict[str, Any]:
    import torch
    from .rope import rotate_pre_rope_k

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
    from .rope import capture_pre_rope

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


def _chunk_tensor_bytes(chunk: EncodedChunk | None) -> int:
    if chunk is None:
        return 0
    return sum(_tensor_nbytes(tensor) for tensor in chunk.kv_by_layer.values())


def _tensor_nbytes(tensor: Any) -> int:
    nbytes = getattr(tensor, "nbytes", None)
    if nbytes is not None:
        return int(nbytes)
    element_size = getattr(tensor, "element_size", None)
    nelement = getattr(tensor, "nelement", None)
    if callable(element_size) and callable(nelement):
        return int(element_size() * nelement())
    return 0


def _chunk_map_cpu_bytes(chunks_by_id: dict[str, EncodedChunk]) -> int:
    total = sys.getsizeof(chunks_by_id)
    for chunk_id, chunk in chunks_by_id.items():
        total += sys.getsizeof(chunk_id)
        total += _encoded_chunk_cpu_bytes(chunk)
    return total


def _encoded_chunk_cpu_bytes(chunk: EncodedChunk) -> int:
    total = sys.getsizeof(chunk)
    total += sys.getsizeof(chunk.token_ids)
    total += sum(sys.getsizeof(token_id) for token_id in chunk.token_ids)
    total += sys.getsizeof(chunk.kv_by_layer)
    total += sum(sys.getsizeof(layer_name) for layer_name in chunk.kv_by_layer)
    return total


def _load_hf_model_and_tokenizer(
    *, model: str, dtype: Any, device: str
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    # Chunk token ids must be byte-identical to the ids the vLLM engine sees
    # (the connector matches registered chunks as a prompt prefix), so load
    # the tokenizer through vLLM's resolver rather than AutoTokenizer. With
    # transformers>=4.56 plus mistral_common installed, AutoTokenizer resolves
    # Mistral models to MistralCommonTokenizer while the engine uses the HF
    # fast tokenizer, and the two encode differently from the first token.
    from vllm.transformers_utils.tokenizer import get_tokenizer

    logger.info("Loading pre-RoPE encoder model=%s device=%s", model, device)
    started = time.perf_counter()
    tokenizer = get_tokenizer(model)
    # The encoder never pads (chunks are encoded one sequence at a time); this
    # fixup only keeps HF tokenizers usable elsewhere. vLLM's MistralTokenizer
    # wrapper exposes no pad_token attribute, so skip it there.
    if getattr(tokenizer, "pad_token", None) is None and hasattr(tokenizer, "eos_token"):
        try:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        except AttributeError:
            pass

    hf_model = AutoModelForCausalLM.from_pretrained(
        model,
        dtype=dtype,
        device_map={"": device},
    )
    hf_model.eval()
    logger.info(
        "Loaded pre-RoPE encoder model=%s in %.1fs gpu_gb=%.1f",
        model,
        time.perf_counter() - started,
        torch.cuda.memory_allocated() / 1e9,
    )
    return tokenizer, hf_model
