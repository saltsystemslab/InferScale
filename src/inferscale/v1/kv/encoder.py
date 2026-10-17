"""Pre-RoPE chunk encoding and position-correct composition.

Chunks are encoded once with a Hugging Face forward pass that captures the
keys before rotary embedding is applied and keeps the values as computed.
Composition concatenates the selected chunks and applies RoPE once at their
new contiguous positions, so a chunk's KV is valid at any position in any
composed memory.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

from ..types import EncodingPlan, KVChunk, ScaffoldChunks, ScaffoldTokens
from .rope import capture_pre_rope, extract_cos_sin_from_model, rotate_pre_rope_k
from .tokenization import encode_text_no_special

logger = logging.getLogger(__name__)

# Accepted KV dtype spellings and the torch dtype name each selects.
KV_DTYPE_ALIASES = {
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "float32": "float32",
    "fp32": "float32",
}

# vLLM's name for the attention module of decoder layer ``index``; KV chunks
# are keyed by it so the connector can address the paged cache per layer.
VLLM_ATTENTION_LAYER_NAME = "model.layers.{index}.self_attn.attn"


def torch_dtype(dtype_name: str) -> Any:
    import torch

    try:
        return getattr(torch, KV_DTYPE_ALIASES[dtype_name.lower()])
    except KeyError:
        raise ValueError(f"Unsupported KV dtype: {dtype_name!r}") from None


def attention_layer_name(index: int) -> str:
    return VLLM_ATTENTION_LAYER_NAME.format(index=index)


class ChunkedRopeEncoder:
    """Hugging Face pre-RoPE encoder shared while precomputing chunks."""

    def __init__(self, *, model: str, dtype: str, device: str, max_position: int) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("KV injection requires a CUDA device.")
        tokenizer, hf_model = _load_hf_model_and_tokenizer(
            model=model, dtype=torch_dtype(dtype), device=device
        )
        head_dim = _head_dim(hf_model.config)
        positions = torch.arange(max_position, device=device)
        cos_table, sin_table = extract_cos_sin_from_model(hf_model, positions, head_dim)
        self._init_state(
            model=model,
            device=device,
            max_position=max_position,
            tokenizer=tokenizer,
            hf_model=hf_model,
            cos_table=cos_table,
            sin_table=sin_table,
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
    ) -> ChunkedRopeEncoder:
        """Encoder in the released state, built from cached RoPE tables.

        compose works and encode fails exactly as they do after
        release_model, so cache hits never load the model weights.
        """
        encoder = cls.__new__(cls)
        encoder._init_state(
            model=model,
            device=device,
            max_position=max_position,
            tokenizer=tokenizer,
            hf_model=None,
            cos_table=cos_table,
            sin_table=sin_table,
        )
        return encoder

    def _init_state(
        self,
        *,
        model: str,
        device: str,
        max_position: int,
        tokenizer: Any,
        hf_model: Any,
        cos_table: Any,
        sin_table: Any,
    ) -> None:
        self.model = model
        self.device = device
        self.max_position = max_position
        self.tokenizer = tokenizer
        self.hf_model = hf_model
        self.head_dim = int(cos_table.shape[-1])
        self.cos_table = cos_table
        self.sin_table = sin_table

    def encode(self, chunk_id: str, token_ids: list[int]) -> KVChunk:
        if not token_ids:
            raise RuntimeError(f"Chunk {chunk_id} tokenized to zero tokens.")
        if len(token_ids) > self.max_position:
            raise RuntimeError(
                f"Chunk {chunk_id} has {len(token_ids)} tokens, "
                f"exceeding max_position={self.max_position}."
            )
        kv_by_layer = self._encode_span(token_ids, slice_start=0, slice_end=len(token_ids))
        return KVChunk(chunk_id=chunk_id, token_ids=list(token_ids), kv_by_layer=kv_by_layer)

    def encode_plan(self, plan: EncodingPlan) -> KVChunk:
        """Encode context plus target and keep only the target span's KV."""
        kv_by_layer = self._encode_span(
            plan.input_token_ids, slice_start=plan.slice_start, slice_end=plan.slice_end
        )
        return KVChunk(
            chunk_id=plan.chunk_id,
            token_ids=list(plan.target_token_ids),
            kv_by_layer=kv_by_layer,
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
        """Unload the model weights while keeping the RoPE tables available."""
        self.hf_model = None
        self.tokenizer = None
        _free_cuda_memory()

    def close(self) -> None:
        self.release_model()
        self.cos_table = None
        self.sin_table = None
        _free_cuda_memory()

    def _encode_span(
        self, token_ids: list[int], *, slice_start: int, slice_end: int
    ) -> dict[str, Any]:
        if self.hf_model is None:
            raise RuntimeError("Cannot encode KV chunk because the HF encoder has been released.")
        kv_by_layer = _encode_token_chunk_pre_rope(
            self.hf_model, token_ids, slice_start=slice_start, slice_end=slice_end
        )
        return {
            layer_name: tensor.detach().to(device=self.device, non_blocking=True).contiguous()
            for layer_name, tensor in kv_by_layer.items()
        }


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


def require_tokenizer_parity(encoder_probe_token_ids: list[int], engine_tokenizer: Any) -> None:
    """The connector matches chunk ids as a prompt prefix, so the encoder and
    engine tokenizers must produce identical ids."""
    engine_probe_token_ids = encode_text_no_special(engine_tokenizer, TOKENIZER_PARITY_PROBE_TEXT)
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
    chunks: list[KVChunk], *, device: str, max_position: int, cos_table: Any, sin_table: Any
) -> dict[str, Any]:
    import torch

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
    composed: dict[str, Any] = {}
    for layer_name in chunks[0].kv_by_layer:
        layer_chunks = [
            chunk.kv_by_layer[layer_name].to(device=device, non_blocking=True) for chunk in chunks
        ]
        k_pre = torch.cat([chunk_kv[0] for chunk_kv in layer_chunks], dim=0)
        values = torch.cat([chunk_kv[1] for chunk_kv in layer_chunks], dim=0)
        # Keys are stored as [tokens, heads, head_dim]; RoPE rotates per head
        # over the token axis, so rotate the transposed view.
        k_rot = rotate_pre_rope_k(
            k_pre.transpose(0, 1).contiguous(), cos_table[:total_tokens], sin_table[:total_tokens]
        )
        composed[layer_name] = torch.stack([k_rot.transpose(0, 1).contiguous(), values], dim=0)
    return composed


def _encode_token_chunk_pre_rope(
    hf_model: Any, token_ids: list[int], *, slice_start: int, slice_end: int
) -> dict[str, Any]:
    import torch

    if slice_start < 0 or slice_end < slice_start or slice_end > len(token_ids):
        raise ValueError("Invalid KV slice bounds for pre-RoPE encoding.")
    device = next(hf_model.parameters()).device
    input_ids = torch.tensor([token_ids], device=device)
    with torch.no_grad(), capture_pre_rope() as capture:
        outputs = hf_model(input_ids=input_ids, use_cache=True)
    values = _post_rope_values(outputs.past_key_values)
    if len(capture.keys) != len(values):
        raise RuntimeError(
            f"RoPE capture saw {len(capture.keys)} layers but model has {len(values)}."
        )
    return {
        attention_layer_name(index): torch.stack(
            [
                _slice_tokens(k_pre, slice_start, slice_end),
                _slice_tokens(value, slice_start, slice_end),
            ],
            dim=0,
        )
        for index, (k_pre, value) in enumerate(zip(capture.keys, values))
    }


def _post_rope_values(past_key_values: Any) -> list[Any]:
    """Value tensors per layer; the cache object's layout depends on the transformers version."""
    if hasattr(past_key_values, "layers"):
        return [layer.values for layer in past_key_values.layers]
    if hasattr(past_key_values, "key_cache"):
        return list(past_key_values.value_cache)
    return [values for _, values in past_key_values]


def _slice_tokens(tensor: Any, start: int, end: int) -> Any:
    """``[1, heads, tokens, head_dim]`` to a contiguous ``[tokens, heads, head_dim]`` span."""
    return tensor.squeeze(0)[:, start:end, :].transpose(0, 1).contiguous()


def _head_dim(config: Any) -> int:
    # Older model configs do not spell out head_dim.
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    return int(head_dim)


def _free_cuda_memory() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


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


def _load_hf_model_and_tokenizer(*, model: str, dtype: Any, device: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM

    logger.info("Loading pre-RoPE encoder model=%s device=%s", model, device)
    started = time.perf_counter()
    tokenizer = load_encoder_tokenizer(model)
    hf_model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, device_map={"": device})
    hf_model.eval()
    logger.info(
        "Loaded pre-RoPE encoder model=%s in %.1fs gpu_gb=%.1f",
        model,
        time.perf_counter() - started,
        torch.cuda.memory_allocated() / 1e9,
    )
    return tokenizer, hf_model
