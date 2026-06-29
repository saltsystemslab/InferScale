# SPDX-License-Identifier: Apache-2.0
"""
rope_inject.py — on-the-fly RoPE rotation for pre-RoPE stored K.

These are PURE FUNCTIONS over tensors. No model dependency. Easy to unit-test
against HuggingFace's apply_rotary_pos_emb.

The key entry point at injection time is:

    K_rotated = rotate_pre_rope_k(
        k_pre=stored_k_pre,
        virtual_positions=torch.arange(v0, v0 + n),
        cos_table=cos_full,   # cos for ALL positions in the model's range
        sin_table=sin_full,
    )

Where (cos_full, sin_full) are computed from the model's RoPE base frequency.
We compute these from scratch via compute_rope_cos_sin() so injection-time
rotation does NOT require keeping a model loaded.

NOTE: HF's `apply_rotary_pos_emb` uses the GPT-NeoX-style "rotate_half"
convention: split the last dim in HALVES (not interleaved pairs). We match
that convention here. If you encounter a model family using the
GPT-J-style interleaved convention, this module will need a separate
rotate_pairs() variant; check by comparing rotate_pre_rope_k() output
against the model's apply_rotary_pos_emb on a known input.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# RoPE table computation
# ---------------------------------------------------------------------------


def compute_rope_cos_sin(
    positions: torch.Tensor,    # [N] long
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the standard Llama-family RoPE (cos, sin) for the given positions.

    Returns:
        cos, sin: each [N, head_dim]. The last dimension is the SAME table
        repeated twice ([freqs, freqs] concatenated), matching HF's
        rotate_half convention. So cos[:, :head_dim//2] == cos[:, head_dim//2:].

    NOTE: This is the "vanilla" RoPE, NOT YaRN / NTK-aware / Llama-3.1-style
    interpolated RoPE. For Llama-3.1-8B (which uses the new RoPE scaling),
    you should pull the cos/sin from the model's rotary_emb module rather
    than recomputing here. See `extract_cos_sin_from_model()` below.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (
        torch.arange(0, half, device=device, dtype=torch.float32) / half
    ))
    pos = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)               # [N, half]
    emb = torch.cat([freqs, freqs], dim=-1)          # [N, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def extract_cos_sin_from_model(
    model,
    positions: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pull cos/sin from a loaded HF model's rotary embedding module. Use this
    instead of compute_rope_cos_sin() when the model has non-standard RoPE
    (Llama-3.1's scaled rope, YaRN, etc.).

    The model's rotary_emb is typically at one of:
      model.model.rotary_emb           (Llama-3+ recent transformers)
      model.model.layers[0].self_attn.rotary_emb   (older versions)

    Returns: (cos, sin), each [N, head_dim].
    """
    rotary = None
    for path in ["model.rotary_emb", "model.layers.0.self_attn.rotary_emb"]:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            rotary = obj
            break
        except AttributeError:
            continue
    if rotary is None:
        raise RuntimeError(
            "Could not locate rotary_emb on model. Inspect model and pass "
            "cos/sin explicitly."
        )

    # Modern transformers: rotary_emb takes (x, position_ids), returns (cos, sin)
    # x is just used for dtype/device.
    # Find device by checking parameters first, then buffers, then fall back
    # to the model's overall device. rotary_emb is typically buffer-only.
    device = None
    for p in rotary.parameters():
        device = p.device
        break
    if device is None:
        for b in rotary.buffers():
            device = b.device
            break
    if device is None:
        device = next(model.parameters()).device
    pos_ids = positions.to(device=device, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(
        1, positions.shape[0], head_dim,
        device=device, dtype=torch.float32,
    )
    cos, sin = rotary(dummy, pos_ids)
    # Returned shape: [batch=1, seq, head_dim]. Squeeze.
    return cos.squeeze(0), sin.squeeze(0)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HF GPT-NeoX-style rotate_half: split last dim in halves and swap with
    a sign flip. Matches transformers.models.llama.modeling_llama.rotate_half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def rotate_pre_rope_k(
    k_pre: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to a pre-rotation K tensor.

    Args:
        k_pre: [..., seq, head_dim] — last two dims are seq and head_dim.
               Common shapes: [num_kv, seq, head_dim] (per-layer, batch=1)
                              [seq, num_kv, head_dim]
        cos:   [seq, head_dim] or [batch, seq, head_dim]
        sin:   same shape as cos

    Returns:
        k_rotated: same shape as k_pre

    The function is shape-agnostic over leading dims; it just needs cos/sin
    to broadcast against the (seq, head_dim) trailing dims. Reshape your
    inputs accordingly before calling.
    """
    # Squeeze batch dim from cos/sin if present
    if cos.dim() == 3:
        cos = cos.squeeze(0)
    if sin.dim() == 3:
        sin = sin.squeeze(0)

    # We need cos/sin to broadcast against k_pre's trailing dims (seq, head_dim).
    # If k_pre is [num_kv, seq, head_dim], we want cos broadcast as
    # [1, seq, head_dim]. Add leading singleton dims to match.
    while cos.dim() < k_pre.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    cos = cos.to(dtype=k_pre.dtype, device=k_pre.device)
    sin = sin.to(dtype=k_pre.dtype, device=k_pre.device)

    return (k_pre * cos) + (_rotate_half(k_pre) * sin)


# ---------------------------------------------------------------------------
# Top-level entry point for injection-time use
# ---------------------------------------------------------------------------


def rotate_chunk_at_virtual_position(
    k_pre_chunk: torch.Tensor,    # [num_kv, n, head_dim]   (per layer, batch=1)
    virtual_start: int,
    cos_table: torch.Tensor,      # [P_max, head_dim]  precomputed RoPE cos
    sin_table: torch.Tensor,      # [P_max, head_dim]
) -> torch.Tensor:
    """
    The injection-time rotation step.

    Args:
        k_pre_chunk: pre-RoPE K for ONE chunk, ONE layer.
                     Shape [num_kv_heads, num_chunk_tokens, head_dim].
        virtual_start: integer virtual position v_0. Tokens of this chunk get
                       virtual positions v_0, v_0+1, ..., v_0 + n - 1.
        cos_table, sin_table: precomputed RoPE tables covering positions
                              [0, P_max). Must have P_max >= virtual_start + n.

    Returns:
        k_rotated: same shape as k_pre_chunk, rotated for the chosen virtual
        positions. Drop-in replacement for HF's stored post-RoPE K.

    This function realizes Theorem (i) of the chunked-RoPE design: the chunk
    can be placed at ANY virtual position; the rotation is applied at
    request time. Combined with cross-chunk independence (Theorem iii),
    this means multiple stored chunks can be composed at request time
    without re-encoding.
    """
    n = k_pre_chunk.shape[-2]
    end = virtual_start + n
    if end > cos_table.shape[0]:
        raise ValueError(
            f"Virtual position range [{virtual_start}, {end}) exceeds "
            f"precomputed RoPE table size {cos_table.shape[0]}. Recompute "
            "cos/sin with a larger range."
        )
    cos = cos_table[virtual_start:end]   # [n, head_dim]
    sin = sin_table[virtual_start:end]
    return rotate_pre_rope_k(k_pre_chunk, cos, sin)
