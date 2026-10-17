"""Capture of pre-rotation keys from a Hugging Face forward pass, and RoPE math.

Hugging Face applies rotary position embedding inside each attention layer
through the module-level ``apply_rotary_pos_emb`` of the model family. The
encoder needs the keys from before that rotation so a chunk's KV can be
rotated at its final position inside a composed memory, so the capture
patches that function in every supported family for the duration of one
forward pass and records the key it was handed per layer.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

_ROPE_MODULES = (
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.qwen3.modeling_qwen3",
)


@dataclass(slots=True)
class RoPECapture:
    """Pre-rotation key tensors in layer order for one forward pass."""

    keys: list[Any] = field(default_factory=list)


def _capturing(original: Any, capture: RoPECapture) -> Any:
    def apply_rotary_pos_emb(
        q: Any, k: Any, cos: Any, sin: Any, position_ids: Any = None, *args: Any, **kwargs: Any
    ) -> Any:
        capture.keys.append(k.detach().clone())
        return original(q, k, cos, sin, position_ids, *args, **kwargs)

    return apply_rotary_pos_emb


@contextlib.contextmanager
def capture_pre_rope() -> Iterator[RoPECapture]:
    """Capture pre-RoPE K tensors during one Hugging Face model forward pass."""
    capture = RoPECapture()
    patched: list[tuple[Any, Any]] = []
    for module_path in _ROPE_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        # A family module without the function cannot be captured; skip it.
        original = getattr(module, "apply_rotary_pos_emb", None)
        if original is None:
            continue
        module.apply_rotary_pos_emb = _capturing(original, capture)
        patched.append((module, original))
    if not patched:
        raise RuntimeError(
            "Could not patch apply_rotary_pos_emb in any of: "
            f"{', '.join(_ROPE_MODULES)}. Is the model loaded?"
        )
    try:
        yield capture
    finally:
        for module, original in patched:
            module.apply_rotary_pos_emb = original


def extract_cos_sin_from_model(model: Any, positions: Any, head_dim: int) -> tuple[Any, Any]:
    """Pull RoPE cos and sin tables for ``positions`` from a loaded model."""
    import torch

    rotary = _find_rotary_embedding(model)
    device = _module_device(rotary, model)
    position_ids = positions.to(device=device, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, positions.shape[0], head_dim, device=device, dtype=torch.float32)
    cos, sin = rotary(dummy, position_ids)
    return cos.squeeze(0), sin.squeeze(0)


def _find_rotary_embedding(model: Any) -> Any:
    """The shared rotary module, or the first layer's own one in older families."""
    try:
        return model.model.rotary_emb
    except AttributeError:
        pass
    try:
        return model.model.layers[0].self_attn.rotary_emb
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("Could not locate rotary_emb on model.") from exc


def _module_device(rotary: Any, model: Any) -> Any:
    for tensors in (rotary.parameters(), rotary.buffers(), model.parameters()):
        tensor = next(tensors, None)
        if tensor is not None:
            return tensor.device
    raise RuntimeError("Could not determine the device of the rotary embedding.")


def rotate_pre_rope_k(k_pre: Any, cos: Any, sin: Any) -> Any:
    """Apply RoPE to a pre-rotation K tensor."""
    if cos.dim() == 3:
        cos = cos.squeeze(0)
    if sin.dim() == 3:
        sin = sin.squeeze(0)
    while cos.dim() < k_pre.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.to(dtype=k_pre.dtype, device=k_pre.device)
    sin = sin.to(dtype=k_pre.dtype, device=k_pre.device)
    return (k_pre * cos) + (_rotate_half(k_pre) * sin)


def _rotate_half(value: Any) -> Any:
    import torch

    half = value.shape[-1] // 2
    return torch.cat([-value[..., half:], value[..., :half]], dim=-1)
