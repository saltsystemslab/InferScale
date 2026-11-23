from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.rag.estimate import run_estimate


@pytest.fixture
def architecture(monkeypatch):
    transformers = pytest.importorskip("transformers", reason="AutoConfig supplies the geometry")
    requested: list[str] = []

    def publish(**attributes) -> list[str]:
        def from_pretrained(model: str) -> SimpleNamespace:
            requested.append(model)
            return SimpleNamespace(**attributes)

        monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
        return requested

    return publish


@pytest.mark.parametrize("model,layers,kv_heads", [("Qwen/Qwen2.5-7B-Instruct", 28, 4)])
def test_configured_answer_models_need_no_autoconfig(
    multihop_config, architecture, model, layers, kv_heads
):
    requested = architecture()

    estimate = run_estimate(multihop_config(model=model))

    assert estimate["kv_bytes_per_token"] == layers * kv_heads * 128 * 2 * 2
    assert requested == []


@pytest.mark.parametrize(
    "attributes,geometry",
    [
        (
            {
                "num_hidden_layers": 12,
                "num_key_value_heads": 4,
                "num_attention_heads": 16,
                "head_dim": 64,
                "hidden_size": 1024,
            },
            (12, 4, 64),
        ),
        ({"num_hidden_layers": 6, "num_attention_heads": 8, "hidden_size": 1024}, (6, 8, 128)),
    ],
)
def test_other_models_take_their_geometry_from_the_attributes_their_architecture_config_has(
    multihop_config, architecture, attributes, geometry
):
    requested = architecture(**attributes)
    layers, kv_heads, head_dim = geometry

    estimate = run_estimate(multihop_config(model="another/model"))

    assert estimate["kv_bytes_per_token"] == layers * kv_heads * head_dim * 2 * 2
    assert requested == ["another/model"]
