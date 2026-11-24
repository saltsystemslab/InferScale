from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest
from library_fakes import FakeEmbedder, FakeEngine, FakeIndex, FakeTokenizer
from loguru import logger
from rag_fixtures import write_multihop_files

from benchmarks.common.config import RuntimeConfig
from benchmarks.rag import corpus, embedder, engine_client, kv_precompute, retrieval
from benchmarks.rag.config import RagBenchConfig
from inferscale.v1.types import EncodingPlan, KVChunk

KV_LAYER = "model.layers.0.self_attn.attn"


@pytest.fixture
def rag_log() -> Iterator[list[tuple[str, str]]]:
    entries: list[tuple[str, str]] = []
    sink = logger.add(
        lambda message: entries.append((message.record["level"].name, message.record["message"])),
        level="INFO",
    )
    yield entries
    logger.remove(sink)


@pytest.fixture
def runtime(tmp_path) -> RuntimeConfig:
    return RuntimeConfig.from_dict(
        {"storage": {"runtime_root": str(tmp_path / "runtime")}}, root=tmp_path
    )


@pytest.fixture
def corpus_tokenizer(monkeypatch) -> list[tuple[str, bool]]:
    loads: list[tuple[str, bool]] = []

    def load(model: str, *, allow_transformers_fallback: bool = False) -> FakeTokenizer:
        loads.append((model, allow_transformers_fallback))
        return FakeTokenizer()

    monkeypatch.setattr(corpus, "load_rag_tokenizer", load)
    return loads


@pytest.fixture
def multihop_config(tmp_path, runtime, corpus_tokenizer) -> Callable[..., RagBenchConfig]:
    data_dir = write_multihop_files(tmp_path / "data")

    def build(**settings: Any) -> RagBenchConfig:
        authored = {
            "data_dir": str(data_dir),
            "chunk_size": 48,
            "context_window": 1,
            "top_k": 3,
            "run_id": "run",
            "skip_judge": True,
            "inferscale": {"kv": {"device": "cpu"}},
            **settings,
        }
        return RagBenchConfig.from_dict(authored, runtime=runtime)

    return build


@pytest.fixture
def embedded_batches(monkeypatch) -> list[list[str]]:
    batches: list[list[str]] = []

    class RecordingEmbedder(FakeEmbedder):
        def __init__(self, **_: Any) -> None:
            super().__init__()
            self.document_calls = batches

    monkeypatch.setattr(embedder, "LocalEmbedder", RecordingEmbedder)
    return batches


class JasperIndexDouble(FakeIndex):
    def __init__(self, config: Any = None) -> None:
        super().__init__()
        self.config = config

    def memory_stats(self) -> dict[str, Any]:
        return {"jasper_vector_count": self.vector_count}


@pytest.fixture
def jasper_indexes(monkeypatch) -> list[JasperIndexDouble]:
    built: list[JasperIndexDouble] = []

    def build(config: Any = None) -> JasperIndexDouble:
        built.append(JasperIndexDouble(config))
        return built[-1]

    monkeypatch.setattr(retrieval, "JasperIndex", build)
    return built


@pytest.fixture
def engine(monkeypatch) -> FakeEngine:
    fake_engine = FakeEngine()
    monkeypatch.setattr(engine_client, "VLLMEngine", lambda model, dtype, config: fake_engine)
    monkeypatch.setattr(engine_client, "force_vllm_inprocess_mode", lambda: None)
    return fake_engine


class TickingClock:
    def __init__(self) -> None:
        self._ticks = itertools.count(1)

    def perf_counter(self) -> float:
        return float(next(self._ticks))


@pytest.fixture
def ticking_clock(monkeypatch) -> Callable[[ModuleType], None]:
    def install(module: ModuleType) -> None:
        monkeypatch.setattr(module, "time", TickingClock())

    return install


def kv_of(token_ids: list[int]) -> dict[str, Any]:
    torch = pytest.importorskip("torch", reason="KV chunk files are torch payloads")
    keys_and_values = torch.tensor(token_ids, dtype=torch.float32).reshape(1, -1, 1, 1)
    return {KV_LAYER: keys_and_values.repeat(2, 1, 1, 2)}


class RopeEncoderDouble:
    def __init__(self, *, model: str, dtype: str, device: str, max_position: int) -> None:
        torch = pytest.importorskip("torch", reason="KV chunk files are torch payloads")
        self.settings = {
            "model": model,
            "dtype": dtype,
            "device": device,
            "max_position": max_position,
        }
        self.cos_table = torch.ones(4, 2)
        self.sin_table = torch.zeros(4, 2)
        self.encoded_plans: list[EncodingPlan] = []
        self.encoded_scaffold: dict[str, list[int]] = {}
        self.closed = False

    def encode(self, chunk_id: str, token_ids: list[int]) -> KVChunk:
        self.encoded_scaffold[chunk_id] = list(token_ids)
        return KVChunk(chunk_id=chunk_id, token_ids=list(token_ids), kv_by_layer=kv_of(token_ids))

    def encode_plan(self, plan: EncodingPlan) -> KVChunk:
        self.encoded_plans.append(plan)
        return KVChunk(
            chunk_id=plan.chunk_id,
            token_ids=list(plan.target_token_ids),
            kv_by_layer=kv_of(plan.target_token_ids),
            context_ids=plan.context_ids,
            context_prefix_tokens=len(plan.context_token_ids),
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def rope_encoders(monkeypatch) -> list[RopeEncoderDouble]:
    built: list[RopeEncoderDouble] = []

    class RecordedRopeEncoder(RopeEncoderDouble):
        def __init__(self, **settings: Any) -> None:
            super().__init__(**settings)
            built.append(self)

    monkeypatch.setattr(kv_precompute, "ChunkedRopeEncoder", RecordedRopeEncoder)
    return built
