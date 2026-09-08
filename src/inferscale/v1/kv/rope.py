from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass, field
from typing import Any, Iterator


_ROPE_MODULES = (
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.qwen3.modeling_qwen3",
)


@dataclass(slots=True)
class _CaptureSlot:
    k_pre: Any | None = None
    cos: Any | None = None
    sin: Any | None = None
    position_ids: Any | None = None


@dataclass(slots=True)
class RoPECapture:
    layers: list[_CaptureSlot] = field(default_factory=list)

    def append(self, k_pre: Any, cos: Any, sin: Any, position_ids: Any) -> None:
        self.layers.append(
            _CaptureSlot(
                k_pre=k_pre.detach().clone(),
                cos=cos.detach().clone() if cos is not None else None,
                sin=sin.detach().clone() if sin is not None else None,
                position_ids=position_ids.detach().clone() if position_ids is not None else None,
            )
        )


@contextlib.contextmanager
def capture_pre_rope() -> Iterator[RoPECapture]:
    """Capture pre-RoPE K tensors during one HuggingFace model forward pass."""
    capture = RoPECapture()
    patched: list[tuple[Any, Any]] = []

    for module_path in _ROPE_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        if not hasattr(module, "apply_rotary_pos_emb"):
            continue

        original = module.apply_rotary_pos_emb

        def _make_wrapper(original_fn: Any) -> Any:
            def wrapper(
                q: Any,
                k: Any,
                cos: Any,
                sin: Any,
                position_ids: Any = None,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                capture.append(k, cos, sin, position_ids)
                return original_fn(q, k, cos, sin, position_ids, *args, **kwargs)

            return wrapper

        module.apply_rotary_pos_emb = _make_wrapper(original)
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
    """Pull RoPE cos/sin tables from a loaded HuggingFace model."""
    import torch

    rotary = None
    for path in ("model.rotary_emb", "model.layers.0.self_attn.rotary_emb"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            rotary = obj
            break
        except AttributeError:
            continue
    if rotary is None:
        raise RuntimeError("Could not locate rotary_emb on model.")

    device = None
    for param in rotary.parameters():
        device = param.device
        break
    if device is None:
        for buffer in rotary.buffers():
            device = buffer.device
            break
    if device is None:
        device = next(model.parameters()).device

    position_ids = positions.to(device=device, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(
        1,
        positions.shape[0],
        head_dim,
        device=device,
        dtype=torch.float32,
    )
    cos, sin = rotary(dummy, position_ids)
    return cos.squeeze(0), sin.squeeze(0)


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
    left = value[..., :half]
    right = value[..., half:]
    return torch.cat([-right, left], dim=-1)
