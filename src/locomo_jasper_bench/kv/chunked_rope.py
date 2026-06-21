from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any

from ..data import ConversationSample, Turn, format_turn_for_memory
from ..prompts import RETRIEVAL_ANSWER_SYSTEM_PROMPT
from ..vector_types import SearchHit
from .submodule import require_ai_memory_submodule

MEMORY_CONTEXT_LABEL = "Retrieved memory context:\n"
MEMORY_TEMPLATE_PLACEHOLDER = "__LOCOMO_JASPER_RETRIEVED_MEMORY__"
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


def hit_turn_id(hit: SearchHit) -> str | None:
    metadata = hit.payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("turn_id"):
        return str(metadata["turn_id"])
    if hit.payload.get("turn_id"):
        return str(hit.payload["turn_id"])
    return None


def selected_turn_ids(hits: list[SearchHit]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        turn_id = hit_turn_id(hit)
        if turn_id and turn_id not in seen:
            ids.append(turn_id)
            seen.add(turn_id)
    return ids


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
        composition_mode: str = "chunked",
    ) -> None:
        require_ai_memory_submodule()
        import torch
        from encode_memories_pre_rope import PreRoPEMemoryEncoder
        from rope_inject import extract_cos_sin_from_model

        if not torch.cuda.is_available():
            raise RuntimeError("KV injection requires a CUDA device.")
        if composition_mode not in {"chunked", "contiguous"}:
            raise ValueError(f"Unsupported KV composition mode: {composition_mode!r}")

        self.model = model
        self.device = device
        self.max_position = max_position
        self.composition_mode = composition_mode
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
        self.turn_text_by_id: dict[str, str] = {}
        self.precomposed: dict[tuple[str, ...], ComposedMemory] = {}
        self.system_header_chunk, self.system_footer_chunk = self._encode_memory_template_chunks()

    def encode_sample(self, sample: ConversationSample, turn_ids: set[str] | None = None) -> None:
        turns = sample.turns
        if turn_ids is not None:
            turns = [turn for turn in sample.turns if turn.id in turn_ids]
        logger.info(
            "Pre-RoPE encoding %d memory chunks for sample_id=%s mode=%s",
            len(turns),
            sample.sample_id,
            self.composition_mode,
        )
        for turn in turns:
            text = format_turn_for_memory(turn).strip() + "\n"
            self.turn_text_by_id[turn.id] = text
            if self.composition_mode == "chunked":
                self.chunks[turn.id] = self._encode_text_chunk(turn.id, text)
        logger.info("Pre-RoPE encoded %d chunks for sample_id=%s", len(self.chunks), sample.sample_id)

    def precompose_contiguous(self, hits_by_question: list[list[SearchHit]]) -> None:
        if self.composition_mode != "contiguous":
            return

        unique_turn_orders: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for hits in hits_by_question:
            turn_order = tuple(selected_turn_ids(hits))
            if not turn_order:
                raise RuntimeError("Cannot precompose contiguous KV memory because retrieval returned no turn ids.")
            if turn_order not in seen:
                unique_turn_orders.append(turn_order)
                seen.add(turn_order)

        logger.info("Precomposing %d contiguous KV memory prefixes", len(unique_turn_orders))
        for turn_order in unique_turn_orders:
            self.precomposed[turn_order] = self._compose_contiguous_turn_order(turn_order)

    def compose(self, hits: list[SearchHit]) -> ComposedMemory:
        started = time.perf_counter()
        turn_ids = selected_turn_ids(hits)
        if not turn_ids:
            raise RuntimeError("Cannot compose KV memory because retrieval returned no turn ids.")
        if self.composition_mode == "contiguous":
            key = tuple(turn_ids)
            composed = self.precomposed.get(key)
            if composed is None:
                raise RuntimeError("Contiguous KV memory was not precomposed for the retrieved turn order.")
            return ComposedMemory(
                kv_by_layer=composed.kv_by_layer,
                token_ids=composed.token_ids,
                num_tokens=composed.num_tokens,
                compose_time_ms=(time.perf_counter() - started) * 1000,
                selected_turn_ids=composed.selected_turn_ids,
            )

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

        chunks = [self.system_header_chunk, *selected, self.system_footer_chunk]
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

        for attr in (
            "encoder",
            "hf_model",
            "tokenizer",
            "cos_table",
            "sin_table",
            "chunks",
            "turn_text_by_id",
            "precomposed",
            "system_header_chunk",
            "system_footer_chunk",
            "memory_system_header_text",
            "memory_system_footer_text",
        ):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()

    def _encode_turn_chunk(self, turn: Turn) -> EncodedChunk:
        return self._encode_text_chunk(turn.id, format_turn_for_memory(turn).strip() + "\n")

    def _encode_memory_template_chunks(self) -> tuple[EncodedChunk, EncodedChunk]:
        header_tokens, footer_tokens, header_text, footer_text = encode_memory_system_template(self.tokenizer)
        self.memory_system_header_text = header_text
        self.memory_system_footer_text = footer_text
        return (
            self._encode_token_chunk("__system_header__", header_tokens, header_text),
            self._encode_token_chunk("__system_footer__", footer_tokens, footer_text),
        )

    def _encode_text_chunk(self, turn_id: str, text: str) -> EncodedChunk:
        token_ids = encode_text_without_special_tokens(self.tokenizer, text)
        return self._encode_token_chunk(turn_id, token_ids, text)

    def _encode_token_chunk(self, turn_id: str, token_ids: list[int], text: str) -> EncodedChunk:
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

    def _compose_contiguous_turn_order(self, turn_ids: tuple[str, ...]) -> ComposedMemory:
        missing = [turn_id for turn_id in turn_ids if turn_id not in self.turn_text_by_id]
        if missing:
            raise RuntimeError(
                "Retrieved memory chunks were not available for contiguous composition: "
                + ", ".join(missing[:5])
            )

        memory_lines = [
            f"{rank}. {self.turn_text_by_id[turn_id].strip()}"
            for rank, turn_id in enumerate(turn_ids, start=1)
        ]
        memory_text = "\n".join(memory_lines) + "\n"
        full_text = self.memory_system_header_text + memory_text + self.memory_system_footer_text
        token_ids = encode_text_without_special_tokens(self.tokenizer, full_text)
        chunk = self._encode_token_chunk("__contiguous_memory__", token_ids, full_text)
        kv_by_layer = self._compose_chunks([chunk])
        return ComposedMemory(
            kv_by_layer=kv_by_layer,
            token_ids=token_ids,
            num_tokens=len(token_ids),
            compose_time_ms=0.0,
            selected_turn_ids=list(turn_ids),
        )


def memory_system_content() -> str:
    return f"{RETRIEVAL_ANSWER_SYSTEM_PROMPT}\n\n{MEMORY_CONTEXT_LABEL}{MEMORY_TEMPLATE_PLACEHOLDER}"


def split_memory_system_template(tokenizer: Any) -> tuple[str, str]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        rendered = apply_chat_template(
            [{"role": "system", "content": memory_system_content()}],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        rendered = f"SYSTEM: {memory_system_content()}\n"

    if MEMORY_TEMPLATE_PLACEHOLDER not in rendered:
        raise RuntimeError(
            "Could not split memory system prompt because the tokenizer chat template "
            "did not preserve the memory placeholder."
        )
    return rendered.split(MEMORY_TEMPLATE_PLACEHOLDER, 1)


def encode_memory_system_template(tokenizer: Any) -> tuple[list[int], list[int], str, str]:
    header_text, footer_text = split_memory_system_template(tokenizer)
    header_tokens = encode_text_without_special_tokens(tokenizer, header_text)
    footer_tokens = encode_text_without_special_tokens(tokenizer, footer_text)
    if not header_tokens or not footer_tokens:
        raise RuntimeError(
            f"Memory system template produced empty header/footer tokens: "
            f"header={len(header_tokens)} footer={len(footer_tokens)}."
        )
    return header_tokens, footer_tokens, header_text, footer_text


def encode_text_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Tokenizer has no encode method.")
    try:
        return list(encode(text, add_special_tokens=False))
    except TypeError:
        return list(encode(text))


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
