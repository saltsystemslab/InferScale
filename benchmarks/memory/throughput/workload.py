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
        user_rng = random.Random(f"{seed}:requests:{user_index}")
        requests.extend(
            LocomoRequest(
                user_index=user_index,
                sample_id=sample.sample_id,
                question_id=qa.question_id,
                query=qa.question,
            )
            for qa in _draw_questions(sample.qa, requests_per_user, user_rng)
        )
    return requests


def _draw_questions(
    questions: Sequence[QuestionAnswer], count: int, rng: random.Random
) -> list[QuestionAnswer]:
    has_enough_distinct_questions = count <= len(questions)
    if has_enough_distinct_questions:
        return rng.sample(questions, count)
    return rng.choices(questions, k=count)


def sample_for_user(samples: Sequence[ConversationSample], user_index: int) -> ConversationSample:
    return samples[user_index % len(samples)]


def samples_for_users(
    samples: Sequence[ConversationSample], num_users: int
) -> list[ConversationSample]:
    users_with_distinct_samples = min(num_users, len(samples))
    return [
        sample_for_user(samples, user_index) for user_index in range(users_with_distinct_samples)
    ]


def group_requests_by_user(
    requests: Sequence[LocomoRequest], num_users: int
) -> dict[int, list[LocomoRequest]]:
    requests_by_user: dict[int, list[LocomoRequest]] = {
        user_index: [] for user_index in range(num_users)
    }
    for request in requests:
        requests_by_user[request.user_index].append(request)
    return requests_by_user


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
    prompt = build_kv_equivalence_prompt_token_ids(
        tokenizer, memory_token_ids, request_question_answer(request)
    )
    return prompt.prompt_token_ids


def is_progress_step(done: int, total: int) -> bool:
    return done % PROGRESS_LOG_EVERY_USERS == 0 or done == total
