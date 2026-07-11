from __future__ import annotations

import pytest

from locomo_jasper_bench.data import ConversationSample, QuestionAnswer
from locomo_jasper_bench.throughput.workload import build_locomo_requests


def _samples(question_counts: list[int]) -> list[ConversationSample]:
    return [
        ConversationSample(
            sample_id=f"sample-{sample_index}",
            turns=[],
            qa=[
                QuestionAnswer(
                    sample_id=f"sample-{sample_index}",
                    question_id=f"s{sample_index}-q{question_index}",
                    question=f"Question {question_index} about sample {sample_index}?",
                    answer="",
                    category="1",
                )
                for question_index in range(count)
            ],
            raw={},
        )
        for sample_index, count in enumerate(question_counts)
    ]


def test_locomo_requests_assign_conversations_round_robin() -> None:
    samples = _samples([5, 5, 5])

    requests = build_locomo_requests(samples, num_users=7, requests_per_user=2, seed=42)

    assert len(requests) == 14
    by_user = {request.user_index: request.sample_id for request in requests}
    assert by_user[0] == "sample-0"
    assert by_user[1] == "sample-1"
    assert by_user[2] == "sample-2"
    assert by_user[3] == "sample-0"
    assert by_user[6] == "sample-0"
    for request in requests:
        assert request.query.endswith(f"sample {request.sample_id[-1]}?")


def test_locomo_requests_are_deterministic_and_prefix_stable() -> None:
    samples = _samples([30, 30])

    small = build_locomo_requests(samples, num_users=2, requests_per_user=3, seed=42)
    again = build_locomo_requests(samples, num_users=2, requests_per_user=3, seed=42)
    large = build_locomo_requests(samples, num_users=5, requests_per_user=3, seed=42)

    assert small == again
    assert small == large[: len(small)]


def test_locomo_requests_replicas_of_one_sample_ask_different_questions() -> None:
    samples = _samples([50])

    requests = build_locomo_requests(samples, num_users=2, requests_per_user=3, seed=42)
    first_user = {request.question_id for request in requests if request.user_index == 0}
    second_user = {request.question_id for request in requests if request.user_index == 1}

    assert first_user != second_user


def test_locomo_requests_reject_bad_inputs() -> None:
    samples = _samples([2])

    with pytest.raises(ValueError, match="At least one LoCoMo sample"):
        build_locomo_requests([], num_users=1, requests_per_user=1, seed=42)
    with pytest.raises(ValueError, match="num_users"):
        build_locomo_requests(samples, num_users=0, requests_per_user=1, seed=42)
    with pytest.raises(RuntimeError, match="has no questions"):
        build_locomo_requests(
            _samples([0]),
            num_users=1,
            requests_per_user=1,
            seed=42,
        )

    oversampled = build_locomo_requests(samples, num_users=1, requests_per_user=5, seed=42)
    assert len(oversampled) == 5  # falls back to sampling with replacement
