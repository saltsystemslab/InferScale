from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.memory.throughput.batches import PreparedRequest, prepare_user_batches
from benchmarks.memory.throughput.stores import SampleStores
from benchmarks.memory.throughput.workload import LocomoRequest


def _request(user_index: int, sample_id: str) -> LocomoRequest:
    return LocomoRequest(
        user_index=user_index,
        sample_id=sample_id,
        question_id=f"q-{user_index}",
        query=f"query-{user_index}",
    )


def test_batches_charge_each_request_to_every_count_that_includes_its_user() -> None:
    samples = [SimpleNamespace(sample_id="a"), SimpleNamespace(sample_id="b")]
    stores = SampleStores(build_time_s={"a": 1.0, "b": 2.0})
    facts = {"a": ("f1", "f2"), "b": ("f3",)}
    requests_by_user = {0: [_request(0, "a")], 1: [_request(1, "b")], 2: [_request(2, "a")]}
    prepared: list[tuple[str, str]] = []

    def prepare(sample, request):
        prepared.append((sample.sample_id, request.query))
        return PreparedRequest(
            prompt_token_ids=[request.user_index],
            retrieval_s=0.1,
            search_s=0.01,
            prompt_build_s=0.001,
            cached_prefix_tokens=10 * request.user_index,
        )

    batches = prepare_user_batches(
        condition="test",
        samples=samples,
        stores=stores,
        facts_by_sample=facts,
        user_counts=(1, 3),
        requests_by_user=requests_by_user,
        prepare_request=prepare,
    )

    # Every request is prepared exactly once, in user order.
    assert prepared == [("a", "query-0"), ("b", "query-1"), ("a", "query-2")]
    one, three = batches
    assert one.num_users == 1 and three.num_users == 3
    assert one.prompts == [{"prompt_token_ids": [0]}]
    assert one.cached_prefix_tokens == [0]
    assert one.fact_count_total == 2
    assert one.memory_setup_time_s == 1.0
    assert three.prompts == [
        {"prompt_token_ids": [0]},
        {"prompt_token_ids": [1]},
        {"prompt_token_ids": [2]},
    ]
    assert three.cached_prefix_tokens == [0, 10, 20]
    assert three.fact_count_total == 5
    assert three.memory_setup_time_s == 3.0
    assert three.retrieval_time_s == pytest.approx(0.3)
    assert three.vector_search_time_s == pytest.approx(0.03)
