from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from ..data import ConversationSample
from ..kv.prompting import tokenize_messages


@dataclass(frozen=True, slots=True)
class LocomoRequest:
    user_id: str
    user_index: int
    sample_id: str
    question_id: str
    query: str


def build_locomo_requests(
    samples: list[ConversationSample],
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
        sample = samples[user_index % len(samples)]
        if not sample.qa:
            raise RuntimeError(f"LoCoMo sample {sample.sample_id} has no questions.")
        rng = random.Random(f"{seed}:requests:{user_index}")
        if requests_per_user <= len(sample.qa):
            chosen = rng.sample(sample.qa, requests_per_user)
        else:
            chosen = rng.choices(sample.qa, k=requests_per_user)
        for qa in chosen:
            requests.append(
                LocomoRequest(
                    user_id=user_id(user_index),
                    user_index=user_index,
                    sample_id=sample.sample_id,
                    question_id=qa.question_id,
                    query=qa.question,
                )
            )
    return requests


def build_no_memory_prompt(tokenizer: Any, query: str) -> list[int]:
    return tokenize_messages(
        tokenizer,
        [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": query},
        ],
    )


def user_id(user_index: int) -> str:
    return f"user_{user_index:04d}"
