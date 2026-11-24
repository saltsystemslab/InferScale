from __future__ import annotations

import gc
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from throughput_config import make_throughput_config

from benchmarks.memory.throughput import kv_requests
from benchmarks.memory.throughput.kv_requests import (
    RequestContext,
    RequestTimings,
    SampleRetrieval,
    compose_request,
)
from benchmarks.memory.throughput.stores import StoreSearch
from benchmarks.memory.throughput.workload import LocomoRequest
from inferscale.v1.kv.registry import drop_namespace, get_gpu_memory_store, register_user_memory
from inferscale.v1.types import KVChunk

SAMPLE = SimpleNamespace(sample_id="sample-0")
MEMORY_TOKENS = [1, 2, 5, 6, 7, 9]
TOKENIZER = object()
SCAFFOLD = object()
CHUNK_STORE = object()
ROW_MAP = object()
MEMORY = object()
STABLE_IDS = object()
DEVICE_RESULT = SimpleNamespace(stable_ids=STABLE_IDS)


def chunk(chunk_id: str, token_ids: list[int]) -> KVChunk:
    return KVChunk(chunk_id=chunk_id, token_ids=token_ids, kv_by_layer={})


FACT_CHUNKS = {"f1": chunk("f1", [5]), "f2": chunk("f2", [6, 7])}


def request(user_index: int) -> LocomoRequest:
    return LocomoRequest(user_index, "sample-0", f"q{user_index}", f"query{user_index}")


class Phases:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, backend: str) -> None:
        self.events: list[Any] = []
        self.registered: list[dict[str, Any]] = []
        self.config = make_throughput_config(tmp_path, kv_store_backend=backend, top_k=7)
        self.composer = SimpleNamespace(
            header_chunk=chunk("__header__", [1, 2]),
            footer_chunk=chunk("__footer__", [9]),
            chunks=FACT_CHUNKS,
            encoder=SimpleNamespace(compose=self.compose),
        )
        patches = {
            "synchronize_cuda": lambda: self.events.append("synchronize_cuda"),
            "search_store_for_kv": self.search,
            "fetch_device_chunk": self.fetch_device_chunk,
            "fetch_device_chunks": self.fetch_device_chunks,
            "release_chunks": lambda store, ids: self.events.append(("release", store, list(ids))),
            "register_user_memory": self.register,
            "materialize_store_device_result": self.materialize,
            "build_memory_prompt_token_ids": self.build_prompt_injection_memory,
            "request_prompt_token_ids": self.request_prompt,
        }
        for name, replacement in patches.items():
            monkeypatch.setattr(kv_requests, name, replacement)

    def context(self) -> RequestContext:
        corpus = SimpleNamespace(
            composer=lambda sample_id: self.composer,
            chunk_store=lambda sample_id: CHUNK_STORE,
            tokenizer=TOKENIZER,
        )
        return RequestContext(
            config=self.config,
            corpus=corpus,
            retrieval={"sample-0": SampleRetrieval(memory=MEMORY, row_map=ROW_MAP)},
            namespace="the-namespace",
            scaffold=SCAFFOLD,
        )

    def search(self, memory, query, *, top_k):
        self.events.append(("search", memory, query, top_k))
        return StoreSearch(DEVICE_RESULT, 0.5, 0.125)

    def fetch_device_chunk(self, store, stable_ids, row_map):
        self.events.append(("fetch_device_chunk", store, stable_ids, row_map))
        return chunk("device-selection", [5, 6, 7])

    def fetch_device_chunks(self, store, meta, stable_ids, row_map, *, reverse):
        self.events.append(("fetch_device_chunks", store, meta, stable_ids, row_map, reverse))
        return [meta["f1"], meta["f2"]]

    def compose(self, chunks: list[KVChunk]) -> dict[str, Any]:
        self.events.append(("compose", [item.chunk_id for item in chunks]))
        return {"layer": "composed"}

    def register(self, namespace, **memory):
        self.events.append("register")
        self.registered.append({"namespace": namespace, **memory})

    def materialize(self, memory, device_result):
        self.events.append(("materialize", memory, device_result))
        return ["hit"]

    def build_prompt_injection_memory(self, tokenizer, hits, **options):
        self.events.append(("prompt_injection_memory", tokenizer, hits, options))
        return SimpleNamespace(token_ids=MEMORY_TOKENS)

    def request_prompt(self, tokenizer, memory_token_ids, locomo_request):
        self.events.append(("request_prompt", tokenizer, list(memory_token_ids), locomo_request))
        return [*memory_token_ids, 100]


@pytest.mark.parametrize(
    "backend,selection,selected_ids,released_ids",
    [
        (
            "gpu",
            ("fetch_device_chunk", CHUNK_STORE, STABLE_IDS, ROW_MAP),
            ["device-selection"],
            [],
        ),
        (
            "cpu",
            ("fetch_device_chunks", CHUNK_STORE, FACT_CHUNKS, STABLE_IDS, ROW_MAP, True),
            ["f1", "f2"],
            ["f1", "f2"],
        ),
    ],
)
def test_request_runs_its_phases_in_order_and_times_every_one_of_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ticking_clock,
    backend: str,
    selection: tuple,
    selected_ids: list[str],
    released_ids: list[str],
) -> None:
    phases = Phases(monkeypatch, tmp_path, backend=backend)
    ticking_clock(phases.events)
    timings = RequestTimings()

    prompt_token_ids, memory_token_ids = compose_request(
        phases.context(), SAMPLE, request(0), "request-00000", timings
    )

    assert phases.events == [
        ("search", MEMORY, "query0", 7),
        "synchronize_cuda",
        "clock",
        selection,
        ("compose", ["__header__", *selected_ids, "__footer__"]),
        "synchronize_cuda",
        "clock",
        ("release", CHUNK_STORE, released_ids),
        "clock",
        "register",
        "synchronize_cuda",
        "clock",
        "clock",
        ("materialize", MEMORY, DEVICE_RESULT),
        ("prompt_injection_memory", TOKENIZER, ["hit"], {"memory_scaffold": SCAFFOLD}),
        "clock",
        "clock",
        ("request_prompt", TOKENIZER, MEMORY_TOKENS, request(0)),
        "clock",
    ]
    assert timings == RequestTimings(
        retrieval_s=0.5,
        vector_search_s=0.125,
        compose_s=3.0,
        store_write_s=7.0,
        verify_s=11.0,
        prompt_build_s=15.0,
    )
    assert memory_token_ids == MEMORY_TOKENS
    assert prompt_token_ids == [*MEMORY_TOKENS, 100]
    assert phases.registered == [
        {
            "namespace": "the-namespace",
            "user_id": "request-00000",
            "kv_by_layer": {"layer": "composed"},
            "num_tokens": 6,
            "token_ids": MEMORY_TOKENS,
        }
    ]


def steps(events: list[Any]) -> list[str]:
    return [event if isinstance(event, str) else event[0] for event in events]


def test_compose_time_is_measured_between_cuda_synchronizations_so_it_covers_the_gpu_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ticking_clock
) -> None:
    phases = Phases(monkeypatch, tmp_path, backend="gpu")
    ticking_clock(phases.events)
    timings = RequestTimings()

    compose_request(phases.context(), SAMPLE, request(0), "request-00000", timings)

    compose = steps(phases.events).index("compose")
    assert steps(phases.events)[compose - 3 : compose + 3] == [
        "synchronize_cuda",
        "clock",
        "fetch_device_chunk",
        "compose",
        "synchronize_cuda",
        "clock",
    ]
    assert timings.compose_s == 3.0


def test_cpu_store_looks_rows_up_on_the_gpu_and_stages_only_the_selected_chunks_until_composed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    phases = Phases(monkeypatch, tmp_path, backend="cpu")

    compose_request(phases.context(), SAMPLE, request(0), "request-00000", RequestTimings())

    staged = ("fetch_device_chunks", CHUNK_STORE, FACT_CHUNKS, STABLE_IDS, ROW_MAP, True)
    composed = ("compose", ["__header__", "f1", "f2", "__footer__"])
    released = ("release", CHUNK_STORE, ["f1", "f2"])
    assert [event for event in phases.events if event in (staged, composed, released)] == [
        staged,
        composed,
        released,
    ]
    assert "fetch_device_chunk" not in steps(phases.events)


def test_selected_views_die_with_their_request_so_no_slab_outlives_the_corpus_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    phases = Phases(monkeypatch, tmp_path, backend="gpu")
    views: list[weakref.ref] = []

    class ChunkView:
        chunk_id = "device-selection"
        token_ids = [5, 6, 7]

    def fetch_view(store, stable_ids, row_map):
        view = ChunkView()
        views.append(weakref.ref(view))
        return view

    monkeypatch.setattr(kv_requests, "fetch_device_chunk", fetch_view)

    compose_request(phases.context(), SAMPLE, request(0), "request-00000", RequestTimings())
    gc.collect()

    assert [view() for view in views] == [None]


class TensorOnTheRegistryDevice:
    device = "cuda:0"
    nbytes = 64

    def to(self, **placement: Any) -> Any:
        pytest.fail("the composed memory was transferred")

    def contiguous(self) -> TensorOnTheRegistryDevice:
        return self


def test_registering_a_composed_memory_inserts_it_into_the_gpu_registry_without_a_transfer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    phases = Phases(monkeypatch, tmp_path, backend="gpu")
    composed_kv = TensorOnTheRegistryDevice()
    phases.composer.encoder.compose = lambda chunks: {"layer": composed_kv}
    monkeypatch.setattr(kv_requests, "register_user_memory", register_user_memory)
    registry = get_gpu_memory_store("the-namespace", backend="gpu")

    try:
        compose_request(phases.context(), SAMPLE, request(0), "request-00000", RequestTimings())
        registered = registry.get_user_memory("request-00000")
        registered_kv, registered_token_ids = registered.kv_by_layer, registered.token_ids
    finally:
        drop_namespace("the-namespace")

    assert registered_kv == {"layer": composed_kv}
    assert registered_kv["layer"] is composed_kv
    assert registered_token_ids == MEMORY_TOKENS
