from __future__ import annotations

import math
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from inferscale.v1.kv import rope
from inferscale.v1.kv.encoder import (
    TOKENIZER_PARITY_PROBE_TEXT,
    ChunkedRopeEncoder,
    attention_layer_name,
    load_encoder_tokenizer,
)
from inferscale.v1.types import KVChunk

torch = pytest.importorskip("torch")

LAYER = "model.layers.0.self_attn.attn"


def angle_tables(angles: list[float]) -> tuple[Any, Any]:
    table = torch.tensor([[angle, angle] for angle in angles])
    return table.cos(), table.sin()


def table_encoder(angles: list[float], *, max_position: int | None = None) -> ChunkedRopeEncoder:
    cos_table, sin_table = angle_tables(angles)
    return ChunkedRopeEncoder.from_tables(
        model="m",
        device="cpu",
        max_position=len(angles) if max_position is None else max_position,
        tokenizer="tokenizer",
        cos_table=cos_table,
        sin_table=sin_table,
    )


def test_kv_layers_are_named_after_vllm_attention_modules() -> None:
    assert attention_layer_name(0) == LAYER
    assert attention_layer_name(31) == "model.layers.31.self_attn.attn"


def test_composed_keys_are_rotated_per_head_by_their_position_in_the_memory() -> None:
    angles = [0.0, 0.5, 1.0, 1.5]
    encoder = table_encoder(angles)
    first_keys = torch.tensor([[[1.0, 0.0], [0.0, 2.0]], [[3.0, 0.0], [0.0, 4.0]]])
    second_keys = torch.tensor([[[5.0, 0.0], [0.0, 6.0]]])
    first_values = torch.full((2, 2, 2), 7.0)
    second_values = torch.full((1, 2, 2), 8.0)
    first = KVChunk("a", [1, 2], {LAYER: torch.stack([first_keys, first_values])})
    second = KVChunk("b", [3], {LAYER: torch.stack([second_keys, second_values])})

    composed = encoder.compose([second, first])[LAYER]

    def rotated(vector: list[float], angle: float) -> list[float]:
        x, y = vector
        return [
            x * math.cos(angle) - y * math.sin(angle),
            y * math.cos(angle) + x * math.sin(angle),
        ]

    expected_keys = torch.tensor(
        [
            [rotated([5.0, 0.0], 0.0), rotated([0.0, 6.0], 0.0)],
            [rotated([1.0, 0.0], 0.5), rotated([0.0, 2.0], 0.5)],
            [rotated([3.0, 0.0], 1.0), rotated([0.0, 4.0], 1.0)],
        ]
    )
    assert composed.shape == (2, 3, 2, 2)
    assert torch.allclose(composed[0], expected_keys, atol=1e-6)
    assert torch.equal(composed[1], torch.cat([second_values, first_values]))


class LazyLayerView:
    def __init__(self, layers: dict[str, Any]) -> None:
        self.layers = layers

    def __iter__(self) -> Any:
        return iter(self.layers)

    def __getitem__(self, layer_name: str) -> Any:
        return self.layers[layer_name]


def test_compose_reads_layers_from_a_lazy_view_like_from_a_dict() -> None:
    encoder = table_encoder([0.0, 0.5, 1.0])
    first_layers = {LAYER: torch.arange(8.0).reshape(2, 2, 1, 2)}
    second_layers = {LAYER: torch.arange(4.0).reshape(2, 1, 1, 2) + 10}

    from_dicts = encoder.compose(
        [KVChunk("a", [1, 2], first_layers), KVChunk("b", [3], second_layers)]
    )
    from_views = encoder.compose(
        [
            KVChunk("a", [1, 2], LazyLayerView(first_layers)),
            KVChunk("b", [3], LazyLayerView(second_layers)),
        ]
    )

    assert list(from_views) == [LAYER]
    assert torch.equal(from_views[LAYER], from_dicts[LAYER])


def test_parity_probe_mixes_scripts_so_differing_tokenizers_cannot_agree_by_accident() -> None:
    probe = TOKENIZER_PARITY_PROBE_TEXT

    assert "\U0001f642" in probe
    assert "你好" in probe
    assert all(symbol in probe for symbol in "#$@'():;,.-")
    assert any(character.isdigit() for character in probe)
    assert probe.count("\n") == 2
    assert probe.endswith("end.\n")
    assert "  " not in probe


class MistralLikeTokenizer:
    __slots__ = ("eos_token", "eos_token_id")

    def __init__(self) -> None:
        self.eos_token = "</s>"
        self.eos_token_id = 2


@pytest.fixture
def engine_tokenizers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    tokenizers: dict[str, Any] = {}
    loader = types.ModuleType("vllm.transformers_utils.tokenizer")
    loader.get_tokenizer = lambda model: tokenizers[model]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.transformers_utils.tokenizer", loader)
    return tokenizers


def test_tokenizer_that_cannot_take_a_pad_token_is_returned_as_is(
    engine_tokenizers: dict[str, Any],
) -> None:
    engine_tokenizers["mistral"] = MistralLikeTokenizer()
    engine_tokenizers["no-eos"] = SimpleNamespace()

    assert load_encoder_tokenizer("mistral") is engine_tokenizers["mistral"]
    assert not hasattr(engine_tokenizers["mistral"], "pad_token")
    assert load_encoder_tokenizer("no-eos") is engine_tokenizers["no-eos"]
    assert not hasattr(engine_tokenizers["no-eos"], "pad_token")


def test_capture_calls_the_installed_rope_function_the_way_its_attention_layer_does() -> None:
    llama = pytest.importorskip("transformers.models.llama.modeling_llama")
    queries = torch.arange(8.0).reshape(1, 1, 2, 4)
    keys = queries + 1
    cos, sin = torch.full((1, 2, 4), 0.5), torch.full((1, 2, 4), 2.0)
    rotated_queries, rotated_keys = llama.apply_rotary_pos_emb(queries, keys, cos, sin)

    with rope.capture_pre_rope() as capture:
        captured_queries, captured_keys = llama.apply_rotary_pos_emb(queries, keys, cos, sin)

    assert torch.equal(captured_queries, rotated_queries)
    assert torch.equal(captured_keys, rotated_keys)
    assert [captured.tolist() for captured in capture.keys] == [keys.tolist()]
