"""Which LoCoMo question each simulated user asks, and the prompt a request sends."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from benchmarks.memory.data import ConversationSample, QuestionAnswer
from benchmarks.memory.prompting import build_kv_equivalence_prompt_token_ids

PROGRESS_LOG_EVERY_USERS = 10


@dataclass(frozen=True, slots=True)
class LocomoRequest:
    user_index: int
    sample_id: str
    question_id: str
    query: str


def build_locomo_requests(
    samples: Sequence[ConversationSample],
    *,
    num_users: int,
    requests_per_user: int,
    seed: int,
) -> list[LocomoRequest]:
    """Assign each user a LoCoMo conversation round-robin and draw its questions.

    Question selection is seeded per user, so replicas of the same conversation
    ask different questions while runs stay deterministic.
    """
    if not samples:
        raise ValueError("At least one LoCoMo sample is required.")
    if num_users <= 0:
        raise ValueError("num_users must be greater than zero.")
    if requests_per_user <= 0:
        raise ValueError("requests_per_user must be greater than zero.")

    requests: list[LocomoRequest] = []
    for user_index in range(num_users):
        sample = sample_for_user(samples, user_index)
        if not sample.qa:
            raise RuntimeError(f"LoCoMo sample {sample.sample_id} has no questions.")
        rng = random.Random(f"{seed}:requests:{user_index}")
        if requests_per_user <= len(sample.qa):
            chosen = rng.sample(sample.qa, requests_per_user)
        else:
            chosen = rng.choices(sample.qa, k=requests_per_user)
        requests.extend(
            LocomoRequest(
                user_index=user_index,
                sample_id=sample.sample_id,
                question_id=qa.question_id,
                query=qa.question,
            )
            for qa in chosen
        )
    return requests


def sample_for_user(samples: Sequence[ConversationSample], user_index: int) -> ConversationSample:
    return samples[user_index % len(samples)]


def samples_for_users(
    samples: Sequence[ConversationSample], num_users: int
) -> list[ConversationSample]:
    """The distinct samples the first ``num_users`` users retrieve from, in user order."""
    return [sample_for_user(samples, index) for index in range(min(num_users, len(samples)))]


def group_requests_by_user(
    requests: Sequence[LocomoRequest], num_users: int
) -> dict[int, list[LocomoRequest]]:
    grouped: dict[int, list[LocomoRequest]] = {index: [] for index in range(num_users)}
    for request in requests:
        grouped[request.user_index].append(request)
    return grouped


def request_question_answer(request: LocomoRequest) -> QuestionAnswer:
    return QuestionAnswer(
        sample_id=request.sample_id,
        question_id=request.question_id,
        question=request.query,
        answer="",
        category="",
    )


def request_prompt_token_ids(
    tokenizer: Any, memory_token_ids: list[int], request: LocomoRequest
) -> list[int]:
    """The prompt a request sends: the memory stream followed by its question."""
    prompt = build_kv_equivalence_prompt_token_ids(
        tokenizer, memory_token_ids, request_question_answer(request)
    )
    return prompt.prompt_token_ids


def is_progress_step(done: int, total: int) -> bool:
    """Log every tenth user and the last one."""
    return done % PROGRESS_LOG_EVERY_USERS == 0 or done == total
