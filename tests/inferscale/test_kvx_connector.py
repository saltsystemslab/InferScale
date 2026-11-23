from __future__ import annotations

import enum
import importlib
import importlib.util
import logging
import sys
import types
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from inferscale.v1.kv import registry

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is not None,
    reason="Runs the connector against a stand-in for vLLM; the real one is installed.",
)

CONNECTOR_MODULES = ("connector", "connector_inject", "connector_metadata")
BLOCK_SIZE = 4
MEMORY_TOKENS = list(range(100, 110))
PROMPT = [*MEMORY_TOKENS, 1, 2, 3]


class Role(enum.Enum):
    SCHEDULER = 0
    WORKER = 1


class StandInConnectorMetadata:
    pass


class StandInConnectorBase:
    def __init__(self, vllm_config: Any, role: Any) -> None:
        self._connector_metadata: Any = None

    def bind_connector_metadata(self, connector_metadata: Any) -> None:
        self._connector_metadata = connector_metadata

    def _get_connector_metadata(self) -> Any:
        return self._connector_metadata


class StandInMlaMetadata:
    pass


def stand_in_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.warning_once = logger.warning  # type: ignore[attr-defined]
    return logger


def stand_in_vllm_modules() -> dict[str, types.ModuleType]:
    contents = {
        "vllm.distributed.kv_transfer.kv_connector.v1.base": {
            "KVConnectorBase_V1": StandInConnectorBase,
            "KVConnectorMetadata": StandInConnectorMetadata,
            "KVConnectorRole": Role,
        },
        "vllm.logger": {"init_logger": stand_in_logger},
        "vllm.v1.attention.backend": {"AttentionMetadata": object},
        "vllm.v1.core.sched.output": {"SchedulerOutput": object},
        "vllm.model_executor.layers.attention.mla_attention": {
            "MLACommonMetadata": StandInMlaMetadata
        },
    }
    modules = {}
    for name, attributes in contents.items():
        modules[name] = types.ModuleType(name)
        vars(modules[name]).update(attributes)
    return modules


@pytest.fixture(scope="module")
def kv() -> Iterator[SimpleNamespace]:
    kv_package = importlib.import_module("inferscale.v1.kv")
    stand_ins = stand_in_vllm_modules()
    sys.modules.update(stand_ins)
    try:
        yield SimpleNamespace(
            **{
                name: importlib.import_module(f"inferscale.v1.kv.{name}")
                for name in CONNECTOR_MODULES
            }
        )
    finally:
        for name in CONNECTOR_MODULES:
            sys.modules.pop(f"inferscale.v1.kv.{name}", None)
            if hasattr(kv_package, name):
                delattr(kv_package, name)
        for name in stand_ins:
            sys.modules.pop(name, None)
        registry._REGISTRY.clear()


class RecordingStore:
    def __init__(self, memories: dict[str, Any], num_staging_slots: Any = 0) -> None:
        self.memories = memories
        self.num_staging_slots = num_staging_slots
        self.calls: list[tuple[str, str]] = []

    def peek_user_memory(self, user_id: str) -> Any:
        self.calls.append(("peek", user_id))
        return self.memories.get(user_id)

    def get_user_memory(self, user_id: str) -> Any:
        self.calls.append(("get", user_id))
        return self.memories.get(user_id)

    def prefetch_user_to_gpu(self, user_id: str) -> bool:
        self.calls.append(("prefetch", user_id))
        return user_id in self.memories

    def release_staging(self, user_id: str) -> None:
        self.calls.append(("release", user_id))


def build_connector(kv: SimpleNamespace, store: RecordingStore, **extra_config: Any) -> Any:
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"memory_namespace": "kvx-connector", **extra_config}
        ),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    connector = kv.connector.MemoryKVConnector(vllm_config, Role.SCHEDULER)
    connector._memory_store = store
    kv.connector.reset_load_stats()
    return connector


def scheduler(kv: SimpleNamespace, **extra_config: Any) -> Any:
    memories = {"alice": SimpleNamespace(token_ids=list(MEMORY_TOKENS), num_tokens=10)}
    return build_connector(kv, RecordingStore(memories), **extra_config)


def request(request_id: str = "req-1", prompt: Any = PROMPT, user: Any = "alice") -> Any:
    return SimpleNamespace(
        request_id=request_id, prompt_token_ids=prompt, user=user, sampling_params=None
    )


def step(new: dict[str, list[int]] | None = None, resumed: dict[str, Any] | None = None) -> Any:
    resumed = resumed or {}
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=req_id, block_ids=[block_ids])
            for req_id, block_ids in (new or {}).items()
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=list(resumed),
            resumed_req_ids=list(resumed),
            new_block_ids=[None if ids is None else [ids] for ids in resumed.values()],
        ),
    )


def scheduled(connector: Any, computed_tokens: int, request_id: str = "req-1") -> Any:
    matched = request(request_id)
    external_tokens, _ = connector.get_num_new_matched_tokens(matched, computed_tokens)
    connector.update_state_after_alloc(matched, None, external_tokens)
    return connector


def loads(metadata: Any) -> list[tuple[Any, ...]]:
    return [
        (
            load.user_id,
            load.slot_mapping.tolist(),
            load.num_tokens,
            load.skip_tokens,
            load.request_id,
        )
        for load in metadata.loads
    ]


def slots_of(*block_ids: int) -> list[int]:
    return [block * BLOCK_SIZE + offset for block in block_ids for offset in range(BLOCK_SIZE)]


@pytest.mark.parametrize(("computed_tokens", "tokens_to_load"), [(0, 8), (4, 4)])
def test_memory_hit_reports_the_block_aligned_memory_tokens_still_to_load(
    kv: SimpleNamespace, computed_tokens: int, tokens_to_load: int
) -> None:
    connector = scheduler(kv)

    assert connector.get_num_new_matched_tokens(request(), computed_tokens) == (
        tokens_to_load,
        False,
    )


def test_memory_the_native_prefix_cache_fully_covers_counts_as_covered_without_a_load(
    kv: SimpleNamespace,
) -> None:
    connector = scheduled(scheduler(kv), computed_tokens=8)

    assert loads(connector.build_connector_meta(step(new={"req-1": [10, 11, 12]}))) == []
    assert kv.connector.snapshot_load_stats() == {
        "requests_loaded": 0,
        "tokens_loaded": 0,
        "requests_covered": 1,
    }


def test_matching_reads_token_metadata_and_never_stages_the_memory(kv: SimpleNamespace) -> None:
    connector = scheduler(kv)

    connector.get_num_new_matched_tokens(request(), 0)

    assert connector._memory_store.calls == [("peek", "alice")]


def test_default_user_routes_requests_that_name_no_user(kv: SimpleNamespace) -> None:
    connector = scheduler(kv, default_user_id="alice")

    assert connector.get_num_new_matched_tokens(request(user=None), 0) == (8, False)


@pytest.mark.parametrize(
    ("user", "prompt"),
    [
        ("unknown", PROMPT),
        ("alice", [*MEMORY_TOKENS[:-1], 999, 1, 2]),
    ],
)
def test_user_whose_memory_does_not_prefix_the_prompt_fails_routing(
    kv: SimpleNamespace, user: str, prompt: list[int]
) -> None:
    connector = scheduler(kv)

    with pytest.raises(RuntimeError) as raised:
        connector.get_num_new_matched_tokens(request(user=user, prompt=prompt), 0)

    assert str(raised.value) == f"Memory routing failed: user_id={user!r} request_id='req-1'."


def test_load_after_a_native_prefix_hit_targets_the_blocks_behind_the_cached_ones(
    kv: SimpleNamespace, caplog: pytest.LogCaptureFixture
) -> None:
    connector = scheduled(scheduler(kv), computed_tokens=4)

    with caplog.at_level(logging.INFO, logger="inferscale.v1.kv.connector"):
        caplog.clear()
        metadata = connector.build_connector_meta(step(new={"req-1": [10, 11, 12]}))

    assert loads(metadata) == [("alice", slots_of(11), 4, 4, "req-1")]
    assert [record.getMessage() for record in caplog.records] == [
        "Partial native prefix hit: skipping 4 cached tokens (1 blocks) "
        "before injecting 4 memory tokens for user alice",
        "Built connector metadata for 1 memory loads",
    ]


def test_new_request_that_does_not_expose_its_cached_front_blocks_is_rejected(
    kv: SimpleNamespace,
) -> None:
    connector = scheduled(scheduler(kv), computed_tokens=4)

    with pytest.raises(RuntimeError) as raised:
        connector.build_connector_meta(step(new={"req-1": [11]}))

    assert str(raised.value) == (
        "Request req-1 exposes 1 block(s); expected at least 2 (skip=1, load=1)."
    )


def test_resumed_request_loads_into_its_new_blocks(kv: SimpleNamespace) -> None:
    connector = scheduled(scheduler(kv), computed_tokens=0)

    assert loads(connector.build_connector_meta(step(resumed={"req-1": None}))) == []
    metadata = connector.build_connector_meta(step(resumed={"req-1": [20, 21, 22]}))

    assert loads(metadata) == [("alice", slots_of(20, 21), 8, 0, "req-1")]


def test_resumed_request_after_a_partial_native_prefix_hit_is_rejected(
    kv: SimpleNamespace,
) -> None:
    connector = scheduled(scheduler(kv), computed_tokens=4)

    with pytest.raises(RuntimeError) as raised:
        connector.build_connector_meta(step(resumed={"req-1": [20, 21]}))

    assert str(raised.value) == (
        "Resumed request req-1 has a partial native prefix hit (4 tokens); "
        "this combination is unsupported."
    )


def source_kv(tokens: int, offset: float = 0.0) -> Any:
    values = torch.arange(2 * tokens * 4, dtype=torch.float32) + 1 + offset
    return values.reshape(2, tokens, 1, 4)


def worker_load(user_id: str, num_tokens: int = 1, skip_tokens: int = 0, slots: Any = None) -> Any:
    return SimpleNamespace(
        user_id=user_id,
        num_tokens=num_tokens,
        skip_tokens=skip_tokens,
        request_id=f"req-{user_id}",
        slot_mapping=torch.arange(num_tokens) if slots is None else torch.tensor(slots),
    )


def worker(kv: SimpleNamespace, store: RecordingStore, worker_loads: list[Any]) -> Any:
    connector = build_connector(kv, store, log_memory_hits=False)
    metadata = kv.connector_metadata.MemoryConnectorMetadata()
    metadata.loads.extend(worker_loads)
    connector.bind_connector_metadata(metadata)
    return connector


def forward_context(layers: dict[str, Any], attn_metadata: Any = "attention") -> Any:
    return SimpleNamespace(attn_metadata=attn_metadata, no_compile_layers=layers)


def paged_cache() -> Any:
    return torch.zeros(4, 2, BLOCK_SIZE, 1, 4)


def users_with_one_layer(count: int) -> dict[str, Any]:
    return {
        f"user-{index}": SimpleNamespace(kv_by_layer={"layer.0": source_kv(8)})
        for index in range(count)
    }


def test_decode_only_step_touches_neither_the_store_nor_the_attention_metadata(
    kv: SimpleNamespace,
) -> None:
    store = RecordingStore(users_with_one_layer(1), num_staging_slots=2)

    worker(kv, store, []).start_load_kv(forward_context({}, attn_metadata=None))

    assert store.calls == []


def test_staging_store_prefetches_one_window_and_refills_each_released_slot(
    kv: SimpleNamespace,
) -> None:
    memories = users_with_one_layer(4)
    store = RecordingStore(memories, num_staging_slots=2)
    layers = {"layer.0": SimpleNamespace(kv_cache=paged_cache())}

    worker(kv, store, [worker_load(user) for user in memories]).start_load_kv(
        forward_context(layers)
    )

    assert store.calls == [
        ("prefetch", "user-0"),
        ("prefetch", "user-1"),
        ("get", "user-0"),
        ("release", "user-0"),
        ("prefetch", "user-2"),
        ("get", "user-1"),
        ("release", "user-1"),
        ("prefetch", "user-3"),
        ("get", "user-2"),
        ("release", "user-2"),
        ("get", "user-3"),
        ("release", "user-3"),
    ]
    assert kv.connector.snapshot_load_stats() == {
        "requests_loaded": 4,
        "tokens_loaded": 4,
        "requests_covered": 4,
    }


def test_failed_load_releases_every_staged_memory_exactly_once(kv: SimpleNamespace) -> None:
    memories = users_with_one_layer(4)
    del memories["user-1"]
    store = RecordingStore(memories, num_staging_slots=2)
    layers = {"layer.0": SimpleNamespace(kv_cache=paged_cache())}
    failing = worker(kv, store, [worker_load(f"user-{index}") for index in range(4)])

    with pytest.raises(RuntimeError, match="Memory for user user-1 was not found during load"):
        failing.start_load_kv(forward_context(layers))

    assert store.calls == [
        ("prefetch", "user-0"),
        ("prefetch", "user-1"),
        ("get", "user-0"),
        ("release", "user-0"),
        ("prefetch", "user-2"),
        ("get", "user-1"),
        ("release", "user-1"),
        ("release", "user-2"),
        ("release", "user-3"),
    ]
    assert kv.connector.snapshot_load_stats()["requests_loaded"] == 1


def test_injection_scatters_keys_and_values_into_the_mapped_slots(kv: SimpleNamespace) -> None:
    source = source_kv(8)
    store = RecordingStore({"user-0": SimpleNamespace(kv_by_layer={"layer.0": source})})
    cache = paged_cache()

    worker(kv, store, [worker_load("user-0", num_tokens=3, slots=[5, 6, 12])]).start_load_kv(
        forward_context({"layer.0": SimpleNamespace(kv_cache=cache)})
    )

    assert torch.equal(cache[1, 0, 1:3], source[0, 0:2])
    assert torch.equal(cache[1, 1, 1:3], source[1, 0:2])
    assert torch.equal(cache[3, 0, 0], source[0, 2])
    assert torch.equal(cache[3, 1, 0], source[1, 2])
    assert cache.count_nonzero() == 2 * 3 * 4


def test_layers_without_a_paged_kv_cache_are_skipped(kv: SimpleNamespace) -> None:
    store = RecordingStore({"user-0": SimpleNamespace(kv_by_layer={"layer.0": source_kv(8)})})
    cache = paged_cache()
    layers = {
        "norm": SimpleNamespace(),
        "layer.0": SimpleNamespace(kv_cache=cache),
        "unpaged": SimpleNamespace(kv_cache=None),
    }

    worker(kv, store, [worker_load("user-0", num_tokens=2)]).start_load_kv(forward_context(layers))

    assert cache.count_nonzero() == 2 * 2 * 4


def test_injection_after_a_native_prefix_hit_starts_at_that_offset_into_the_memory(
    kv: SimpleNamespace,
) -> None:
    source = source_kv(8)
    store = RecordingStore({"user-0": SimpleNamespace(kv_by_layer={"layer.0": source})})
    cache = paged_cache()

    worker(
        kv, store, [worker_load("user-0", num_tokens=4, skip_tokens=4, slots=[4, 5, 6, 7])]
    ).start_load_kv(forward_context({"layer.0": SimpleNamespace(kv_cache=cache)}))

    assert torch.equal(cache[1, 0], source[0, 4:8])
    assert torch.equal(cache[1, 1], source[1, 4:8])
    assert cache[0].count_nonzero() == 0


def test_cache_split_into_keys_and_values_on_its_first_axis_is_supported(
    kv: SimpleNamespace,
) -> None:
    source = source_kv(8)
    store = RecordingStore({"user-0": SimpleNamespace(kv_by_layer={"layer.0": source})})
    cache = torch.zeros(2, 3, BLOCK_SIZE, 1, 4)

    worker(kv, store, [worker_load("user-0", num_tokens=2, slots=[1, 6])]).start_load_kv(
        forward_context({"layer.0": SimpleNamespace(kv_cache=cache)})
    )

    assert torch.equal(cache[0, 0, 1], source[0, 0])
    assert torch.equal(cache[0, 1, 2], source[0, 1])
    assert torch.equal(cache[1, 0, 1], source[1, 0])
    assert torch.equal(cache[1, 1, 2], source[1, 1])
    assert cache.count_nonzero() == 2 * 2 * 4


class StagedLayerView:
    def __init__(self, layers: dict[str, Any]) -> None:
        self.layers = layers
        self.waited_for: list[str] = []

    def __iter__(self) -> Iterator[str]:
        return iter(self.layers)

    def __getitem__(self, layer_name: str) -> Any:
        self.waited_for.append(layer_name)
        return self.layers[layer_name]

    def get(self, layer_name: str, default: Any = None) -> Any:
        return self[layer_name] if layer_name in self.layers else default


def test_staged_view_is_probed_through_one_layer_before_the_layers_are_injected(
    kv: SimpleNamespace,
) -> None:
    view = StagedLayerView({"layer.0": source_kv(8), "layer.1": source_kv(8, offset=500)})
    store = RecordingStore({"user-0": SimpleNamespace(kv_by_layer=view)})
    layers = {
        "layer.1": SimpleNamespace(kv_cache=paged_cache()),
        "layer.0": SimpleNamespace(kv_cache=paged_cache()),
    }

    worker(kv, store, [worker_load("user-0")]).start_load_kv(forward_context(layers))

    assert view.waited_for == ["layer.0", "layer.1", "layer.0"]


@pytest.mark.parametrize(
    ("memory_layers", "cache_layers", "message"),
    [
        ({"layer.0": 8}, {"layer.9": "paged"}, "Layer layer.9 not found in memory for user user-0"),
    ],
)
def test_load_that_cannot_fill_every_cache_layer_fails_closed(
    kv: SimpleNamespace, memory_layers: dict[str, int], cache_layers: dict[str, Any], message: str
) -> None:
    memory = SimpleNamespace(
        kv_by_layer={name: source_kv(tokens) for name, tokens in memory_layers.items()}
    )
    layers = {
        name: SimpleNamespace(kv_cache=paged_cache() if cache == "paged" else None)
        for name, cache in cache_layers.items()
    }
    failing = worker(kv, RecordingStore({"user-0": memory}), [worker_load("user-0")])

    with pytest.raises(RuntimeError, match=message):
        failing.start_load_kv(forward_context(layers))

    assert kv.connector.snapshot_load_stats()["requests_loaded"] == 0
