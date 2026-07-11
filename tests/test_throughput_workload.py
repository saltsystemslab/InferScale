from __future__ import annotations

import pytest

from locomo_jasper_bench.data import ConversationSample, QuestionAnswer, Turn
from locomo_jasper_bench.throughput.workload import build_locomo_requests, user_id


def _sample(sample_id: str, question_count: int) -> ConversationSample:
    return ConversationSample(
        sample_id=sample_id,
        turns=[
            Turn(
                sample_id=sample_id,
                session_id="session_1",
                session_index=1,
                turn_index=0,
                speaker="Alice",
                text="I like tea.",
            )
        ],
        qa=[
            QuestionAnswer(
                sample_id=sample_id,
                question_id=f"{sample_id}-q{index}",
                question=f"question {index}",
                answer="",
                category="1",
            )
            for index in range(question_count)
        ],
        raw={},
    )


def test_users_map_to_samples_round_robin() -> None:
    samples = [_sample("sample-a", 4), _sample("sample-b", 4)]

    requests = build_locomo_requests(samples, num_users=5, requests_per_user=1, seed=42)

    assert [request.sample_id for request in requests] == [
        "sample-a",
        "sample-b",
        "sample-a",
        "sample-b",
        "sample-a",
    ]
    assert [request.user_id for request in requests] == [user_id(index) for index in range(5)]


def test_requests_are_deterministic_and_vary_per_replica() -> None:
    samples = [_sample("sample-a", 10)]

    first = build_locomo_requests(samples, num_users=4, requests_per_user=2, seed=42)
    second = build_locomo_requests(samples, num_users=4, requests_per_user=2, seed=42)
    different_seed = build_locomo_requests(samples, num_users=4, requests_per_user=2, seed=7)

    assert [request.question_id for request in first] == [request.question_id for request in second]
    assert [request.question_id for request in first] != [
        request.question_id for request in different_seed
    ]
    per_user = {
        request.user_index: []  # type: ignore[var-annotated]
        for request in first
    }
    for request in first:
        per_user[request.user_index].append(request.question_id)
    assert len({tuple(ids) for ids in per_user.values()}) > 1


def test_requests_per_user_beyond_question_count_uses_replacement() -> None:
    samples = [_sample("sample-a", 2)]

    requests = build_locomo_requests(samples, num_users=1, requests_per_user=5, seed=42)

    assert len(requests) == 5


def test_request_building_rejects_invalid_inputs() -> None:
    samples = [_sample("sample-a", 2)]

    with pytest.raises(ValueError):
        build_locomo_requests([], num_users=1, requests_per_user=1, seed=42)
    with pytest.raises(ValueError):
        build_locomo_requests(samples, num_users=0, requests_per_user=1, seed=42)
    with pytest.raises(ValueError):
        build_locomo_requests(samples, num_users=1, requests_per_user=0, seed=42)
