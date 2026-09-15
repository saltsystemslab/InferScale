"""Pre-RoPE chunk encoding and position-correct composition.

Chunks are encoded once with a Hugging Face forward pass that captures the
keys before rotary embedding is applied and keeps the values as computed.
Composition concatenates the selected chunks and applies RoPE once at their
new contiguous positions, so a chunk's KV is valid at any position in any
composed memory.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..types import EncodingPlan, KVChunk, ScaffoldChunks, ScaffoldTokens
from .rope import extract_cos_sin_from_model
from .tokenization import encode_text_no_special

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
    """Shared HF pre-RoPE encoder used while precomputing chunks."""

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

    @classmethod
    def from_tables(
        cls,
        *,
        model: str,
        device: str,
        max_position: int,
        tokenizer: Any,
        cos_table: Any,
        sin_table: Any,
    ) -> "ChunkedRopeEncoder":
        """Encoder in the post-release_model state, built from cached tables.

        compose works and encode fails closed exactly as they do after
        release_model, so cache hits never load the HF weights.
        """
        encoder = cls.__new__(cls)
        encoder.model = model
        encoder.device = device
        encoder.max_position = max_position
        encoder.tokenizer = tokenizer
        encoder.hf_model = None
        encoder.head_dim = int(cos_table.shape[-1])
        encoder.cos_table = cos_table
        encoder.sin_table = sin_table
        return encoder

    def encode(self, chunk_id: str, token_ids: list[int]) -> KVChunk:
        if self.hf_model is None:
            raise RuntimeError("Cannot encode KV chunk because the HF encoder has been released.")
        if not token_ids:
            raise RuntimeError(f"Chunk {chunk_id} tokenized to zero tokens.")
        if len(token_ids) > self.max_position:
            raise RuntimeError(
                f"Chunk {chunk_id} has {len(token_ids)} tokens, "
                f"exceeding max_position={self.max_position}."
            )
        return KVChunk(
            chunk_id=chunk_id,
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

    def encode_plan(self, plan: EncodingPlan) -> KVChunk:
        """Encode context plus target and keep only the target span's KV."""
        if self.hf_model is None:
            raise RuntimeError("Cannot encode KV chunk because the HF encoder has been released.")
        return KVChunk(
            chunk_id=plan.chunk_id,
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
            context_ids=plan.context_ids,
            context_prefix_tokens=len(plan.context_token_ids),
            raw_context_prefix_tokens=plan.raw_context_tokens,
            context_prefix_truncated_tokens=plan.context_truncated_tokens,
        )

    def compose(self, chunks: list[KVChunk]) -> dict[str, Any]:
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


def encode_scaffold(encoder: ChunkedRopeEncoder, tokens: ScaffoldTokens) -> ScaffoldChunks:
    return ScaffoldChunks(
        header=encoder.encode("scaffold:header", tokens.header_token_ids),
        empty=encoder.encode("scaffold:empty", tokens.empty_token_ids),
        footer=encoder.encode("scaffold:footer", tokens.footer_token_ids),
    )


# Mixed-script probe so tokenizer stacks that differ in template, whitespace,
# byte-fallback, or unicode handling cannot encode it identically by accident.
TOKENIZER_PARITY_PROBE_TEXT = (
    "SPEAKER Caroline (2023-05-08): I'll re-check trip #42 - cost $1,300.50; "
    "email caroline@example.com, emoji \U0001f642, CJK 你好, newline\nend.\n"
)


def require_tokenizer_parity(
    encoder_probe_token_ids: list[int],
    engine_tokenizer: Any,
) -> None:
    """The connector matches chunk ids as a prompt prefix, so the encoder and
    engine tokenizers must produce identical ids."""
    engine_probe_token_ids = encode_text_no_special(
        engine_tokenizer, TOKENIZER_PARITY_PROBE_TEXT
    )
    if encoder_probe_token_ids == engine_probe_token_ids:
        return
    raise RuntimeError(
        "Encoder and engine tokenizers disagree; KV chunk token ids will not "
        "match prompt token ids and injection cannot work. "
        f"encoder_probe_tokens={len(encoder_probe_token_ids)} "
        f"engine_probe_tokens={len(engine_probe_token_ids)} "
        f"engine_tokenizer={type(engine_tokenizer).__name__}. "
        "Ensure ChunkedRopeEncoder loads its tokenizer via "
        "vllm.transformers_utils.tokenizer.get_tokenizer and that the "
        "transformers/vllm/mistral_common versions match the engine's."
    )


def _compose_encoded_chunks(
    chunks: list[KVChunk],
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
            f"Composed memory has {total_tokens} tokens, exceeding max_position={max_position}."
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


def load_encoder_tokenizer(model: str) -> Any:
    """Load the encoder-side tokenizer without touching model weights.

    Chunk token ids must be byte-identical to the ids the vLLM engine sees
    (the connector matches registered chunks as a prompt prefix), so load
    the tokenizer through vLLM's resolver rather than AutoTokenizer. With
    transformers>=4.56 plus mistral_common installed, AutoTokenizer resolves
    Mistral models to MistralCommonTokenizer while the engine uses the HF
    fast tokenizer, and the two encode differently from the first token.
    """
    from vllm.transformers_utils.tokenizer import get_tokenizer

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
    return tokenizer


def _load_hf_model_and_tokenizer(
    *, model: str, dtype: Any, device: str
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    logger.info("Loading pre-RoPE encoder model=%s device=%s", model, device)
    started = time.perf_counter()
    tokenizer = load_encoder_tokenizer(model)

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
