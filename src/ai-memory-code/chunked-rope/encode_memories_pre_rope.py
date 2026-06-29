# SPDX-License-Identifier: Apache-2.0
"""
encode_memories_pre_rope.py — pre-RoPE fork of encode_memories.py.

This is the Path C anchor: encode memory text into KV tensors but capture K
*before* RoPE rotation. The chunked-RoPE design then applies RoPE on the fly
at injection time, with virtual positions chosen at request time.

Strategy
--------
We monkey-patch `transformers.models.llama.modeling_llama.apply_rotary_pos_emb`
(and the equivalent for Mistral / Qwen2 if those modules are imported) for
the duration of the encoder's forward pass. The patched function:

  1. Records q, k as they arrive (pre-rotation), keyed by call order.
  2. Records the (cos, sin) tensors that the model would have applied.
  3. Calls the original, returning the rotated (q, k) so the rest of the
     forward pass proceeds identically — important so we still get a sensible
     V tensor (V is unchanged by RoPE) and for any downstream sanity checks.

This is unrelated to the existing `encode_memories.py`. Output shape and
storage path are identical to the original; the ONLY difference is that
`kv_by_layer[layer_name][0]` (the K tensor) is pre-rotation instead of
post-rotation. We tag the saved metadata so downstream code knows to apply
RoPE on the fly before injection.

Usage
-----
    encoder = PreRoPEMemoryEncoder(model_name="...")
    encoder.load_model()

    kv_by_layer, num_tokens, token_ids, rope_meta = encoder.encode_memory(
        memory_text="..."
    )

    # kv_by_layer is the same shape as the original encoder produces, but K
    # is pre-RoPE. rope_meta carries (cos_table, sin_table, original_positions)
    # for verification and on-the-fly rotation at injection time.

Verification: see test_chunked_rope_phase1_3.py for the round-trip check
that re-rotating with the captured (cos, sin) reproduces HF's post-RoPE K
within bf16 noise.
"""

from __future__ import annotations

import contextlib
import importlib
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
)


# ---------------------------------------------------------------------------
# RoPE capture
# ---------------------------------------------------------------------------

# Modules whose `apply_rotary_pos_emb` we patch. Adding a model family is a
# matter of adding its module path here; the function signature is identical
# across Llama / Mistral / Qwen2 in current transformers.
_ROPE_MODULES = [
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.qwen2.modeling_qwen2",
]


@dataclass
class _CaptureSlot:
    """One layer's worth of pre-RoPE captures. Filled in call-order during
    the forward pass."""

    k_pre: Optional[torch.Tensor] = None      # [batch, num_kv_heads, seq, head_dim]
    cos: Optional[torch.Tensor] = None        # [seq, head_dim] or [batch, seq, head_dim]
    sin: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


@dataclass
class RoPECapture:
    """Holds per-layer pre-RoPE captures for one forward pass."""

    layers: list[_CaptureSlot] = field(default_factory=list)
    _call_idx: int = 0

    def reset(self):
        self.layers.clear()
        self._call_idx = 0

    def append(self, k_pre, cos, sin, position_ids):
        slot = _CaptureSlot(
            k_pre=k_pre.detach().clone(),
            cos=cos.detach().clone() if cos is not None else None,
            sin=sin.detach().clone() if sin is not None else None,
            position_ids=(
                position_ids.detach().clone() if position_ids is not None else None
            ),
        )
        self.layers.append(slot)
        self._call_idx += 1


@contextlib.contextmanager
def capture_pre_rope():
    """Context manager: monkey-patch apply_rotary_pos_emb in every loaded
    model family. On exit, restore the originals.

    Usage:
        with capture_pre_rope() as cap:
            model(input_ids=...)
        # cap.layers is a list of _CaptureSlot, one per attention layer call

    The function signatures vary a bit across transformers versions; we
    detect both the legacy positional form (q, k, cos, sin, position_ids)
    and the new-ish kwarg form (q, k, cos=..., sin=..., position_ids=...).
    """
    cap = RoPECapture()
    patched: list[tuple] = []  # list of (module, original_fn)

    for mod_path in _ROPE_MODULES:
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        if not hasattr(mod, "apply_rotary_pos_emb"):
            continue
        original = mod.apply_rotary_pos_emb

        def _make_wrapper(orig):
            def wrapper(q, k, cos, sin, position_ids=None, *args, **kwargs):
                # Capture pre-RoPE K + the rotation tensors that would be applied.
                # Layer order = call order, which matches the model's layer list
                # for standard decoder-only architectures.
                cap.append(k, cos, sin, position_ids)
                return orig(q, k, cos, sin, position_ids, *args, **kwargs)
            return wrapper

        mod.apply_rotary_pos_emb = _make_wrapper(original)
        patched.append((mod, original))

    if not patched:
        raise RuntimeError(
            "Could not patch apply_rotary_pos_emb in any of: "
            f"{_ROPE_MODULES}. Is the model loaded?"
        )

    try:
        yield cap
    finally:
        for mod, original in patched:
            mod.apply_rotary_pos_emb = original


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class PreRoPEMemoryEncoder:
    """
    Pre-RoPE fork of MemoryEncoder. Same external behavior, except K is
    captured BEFORE rotation, and additional rope_meta is returned so callers
    can apply rotation on the fly at injection time.
    """

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda:0",
        system_prompt_template: str = (
            "Key facts about the user:\n{memory_text}"
        ),
    ):
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.system_prompt_template = system_prompt_template

        self._model: Optional[AutoModelForCausalLM] = None
        self._tokenizer: Optional[AutoTokenizer] = None

    # Identical to MemoryEncoder.load_model except for the imported module
    def load_model(self) -> None:
        if self._model is not None:
            return

        print(f"[pre-RoPE encoder] Loading {self.model_name} ...", flush=True)
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map=self.device,
        )
        self._model.eval()

        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"[pre-RoPE encoder] loaded in {time.time()-t0:.1f}s "
              f"| GPU: {mem_gb:.1f} GB", flush=True)

    # ----------------------------------------------------------------

    def encode_memory(
        self,
        memory_text: str,
        layer_name_map: Optional[dict[int, str]] = None,
    ) -> tuple[dict[str, torch.Tensor], int, list[int], dict]:
        """
        Encode memory text with pre-RoPE K capture.

        Returns:
            (kv_by_layer, num_prefix_tokens, prefix_token_ids, rope_meta)

            kv_by_layer: dict layer_name -> tensor [2, num_tokens, num_kv_heads, head_dim]
                         where dim-0 entry [0] is K_pre (pre-rotation) and
                         entry [1] is V (unchanged by RoPE).
            rope_meta: dict with
                "cos":          [num_tokens, head_dim] cos table at original positions
                "sin":          [num_tokens, head_dim] sin table at original positions
                "position_ids": [num_tokens] original positions used during encoding
                "post_rope_k_check": dict layer_name -> POST-RoPE K from HF
                                      (for the Phase 1 round-trip test)
        """
        assert self._model is not None, "Call load_model() first"
        assert self._tokenizer is not None

        # Replicate the same prefix-detection logic as MemoryEncoder.encode_memory
        system_content = self.system_prompt_template.format(memory_text=memory_text)

        if self._tokenizer.chat_template is None:
            inputs = self._tokenizer(system_content, return_tensors="pt").to(self.device)
            num_prefix = inputs.input_ids.shape[1]
            prefix_ids = inputs.input_ids[0].tolist()
            full_input_ids = inputs.input_ids
        else:
            dummy_a = "What is the user's name?"
            dummy_b = "Tell me about the user's hobbies and interests."
            tokens_a = self._tokenizer.apply_chat_template(
                [{"role": "system", "content": system_content},
                 {"role": "user", "content": dummy_a}],
                tokenize=True, add_generation_prompt=True,
            )
            tokens_b = self._tokenizer.apply_chat_template(
                [{"role": "system", "content": system_content},
                 {"role": "user", "content": dummy_b}],
                tokenize=True, add_generation_prompt=True,
            )
            prefix_len = 0
            for i in range(min(len(tokens_a), len(tokens_b))):
                if tokens_a[i] == tokens_b[i]:
                    prefix_len = i + 1
                else:
                    break
            prefix_ids = tokens_a[:prefix_len]
            num_prefix = prefix_len
            full_input_ids = torch.tensor([tokens_a], device=self.device)

        # Forward pass with RoPE capture active
        with torch.no_grad(), capture_pre_rope() as cap:
            outputs = self._model(input_ids=full_input_ids, use_cache=True)

        # outputs.past_key_values has POST-RoPE K (HF default).
        # cap.layers has PRE-RoPE K + the cos/sin that were applied.
        post_rope_kv = outputs.past_key_values

        if hasattr(post_rope_kv, "layers"):
            post_rope_pairs = [
                (layer.keys, layer.values) for layer in post_rope_kv.layers
            ]
        elif hasattr(post_rope_kv, "key_cache"):
            post_rope_pairs = list(zip(post_rope_kv.key_cache,
                                       post_rope_kv.value_cache))
        else:
            post_rope_pairs = [(k, v) for k, v in post_rope_kv]

        num_layers = len(post_rope_pairs)
        if len(cap.layers) != num_layers:
            raise RuntimeError(
                f"RoPE capture saw {len(cap.layers)} layers but model has "
                f"{num_layers}. Patching may not be hitting all layers."
            )

        # ---- Build kv_by_layer with PRE-RoPE K and (post-rope) V ---------
        # We use V from cap-time as well via the post-rope cache (V is
        # unchanged by RoPE), since the post-rope cache has a clean shape.
        kv_by_layer: dict[str, torch.Tensor] = {}
        post_rope_k_check: dict[str, torch.Tensor] = {}

        for layer_idx, (k_post, v_post) in enumerate(post_rope_pairs):
            slot = cap.layers[layer_idx]
            k_pre = slot.k_pre  # [batch, num_kv_heads, seq, head_dim]

            # Sanity: shapes match between pre and post
            assert k_pre.shape == k_post.shape, (
                f"layer {layer_idx}: k_pre {tuple(k_pre.shape)} != "
                f"k_post {tuple(k_post.shape)}"
            )

            layer_name = (
                layer_name_map[layer_idx]
                if layer_name_map and layer_idx in layer_name_map
                else f"model.layers.{layer_idx}.self_attn.attn"
            )

            # Squeeze batch, truncate to prefix
            k_pre = k_pre.squeeze(0)[:, :num_prefix, :]    # [num_kv, n, d]
            k_post = k_post.squeeze(0)[:, :num_prefix, :]
            v = v_post.squeeze(0)[:, :num_prefix, :]

            # Reshape to [n, num_kv, d] (matches MemoryEncoder format)
            k_pre = k_pre.transpose(0, 1).contiguous()
            k_post = k_post.transpose(0, 1).contiguous()
            v = v.transpose(0, 1).contiguous()

            kv_by_layer[layer_name] = torch.stack([k_pre, v], dim=0)
            post_rope_k_check[layer_name] = k_post

        # ---- rope_meta: cos/sin tables truncated to prefix --------------
        # cos/sin from the model are typically [batch, seq, head_dim] or
        # [seq, head_dim]. Take the first layer's slot since RoPE tables are
        # identical across layers.
        first_slot = cap.layers[0]
        cos = first_slot.cos
        sin = first_slot.sin
        if cos.dim() == 3:    # [batch, seq, head_dim]
            cos = cos[0]
            sin = sin[0]
        cos = cos[:num_prefix].contiguous()
        sin = sin[:num_prefix].contiguous()

        if first_slot.position_ids is not None:
            pos = first_slot.position_ids
            if pos.dim() == 2:
                pos = pos[0]
            pos = pos[:num_prefix].contiguous()
        else:
            pos = torch.arange(num_prefix, device=self.device)

        rope_meta = {
            "cos": cos,
            "sin": sin,
            "position_ids": pos,
            "post_rope_k_check": post_rope_k_check,
        }

        return kv_by_layer, num_prefix, prefix_ids, rope_meta
