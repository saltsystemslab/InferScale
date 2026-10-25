"""The requests of every user count, prepared once and charged to each count that includes them.

A request's prompt and the time its retrieval and prompt construction took are
charged to every user count whose batch includes that user, so one worker
measures all user counts against the same engine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from loguru import logger

from benchmarks.memory.data import ConversationSample
from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.throughput.stores import SampleStores
from benchmarks.memory.throughput.workload import (
    LocomoRequest,
    is_progress_step,
    sample_for_user,
)


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    """One request's prompt and the time its retrieval and prompt construction took.

    ``cached_prefix_tokens`` is the prompt prefix an external KV cache is
    expected to serve; zero for prompt injection.
    """

    prompt_token_ids: list[int]
    retrieval_s: float
    search_s: float
    prompt_build_s: float
    cached_prefix_tokens: int = 0


@dataclass(slots=True)
class UserCountAccumulator:
    """The batch and the setup costs charged to one user count."""

    num_users: int
    memory_setup_time_s: float
    prompts: list[dict[str, list[int]]] = field(default_factory=list)
    cached_prefix_tokens: list[int] = field(default_factory=list)
    retrieval_time_s: float = 0.0
    vector_search_time_s: float = 0.0
    prompt_build_time_s: float = 0.0
    fact_count_total: int = 0

    def includes(self, user_index: int) -> bool:
        return user_index < self.num_users

    def add(self, prepared: PreparedRequest) -> None:
        self.prompts.append({"prompt_token_ids": prepared.prompt_token_ids})
        self.cached_prefix_tokens.append(prepared.cached_prefix_tokens)
        self.retrieval_time_s += prepared.retrieval_s
        self.vector_search_time_s += prepared.search_s
        self.prompt_build_time_s += prepared.prompt_build_s


PrepareRequest = Callable[[ConversationSample, LocomoRequest], PreparedRequest]


def prepare_user_batches(
    *,
    condition: str,
    samples: Sequence[ConversationSample],
    stores: SampleStores,
    facts_by_sample: Mapping[str, Sequence[MemoryFact]],
    user_counts: tuple[int, ...],
    requests_by_user: Mapping[int, Sequence[LocomoRequest]],
    prepare_request: PrepareRequest,
) -> list[UserCountAccumulator]:
    """Prepare every user's requests once, charging each user count that includes the user."""
    batches = [
        UserCountAccumulator(count, stores.setup_time_for_users(samples, count))
        for count in user_counts
    ]
    max_users = len(requests_by_user)
    for user_index in range(max_users):
        sample = sample_for_user(samples, user_index)
        charged = [batch for batch in batches if batch.includes(user_index)]
        for batch in charged:
            batch.fact_count_total += len(facts_by_sample[sample.sample_id])
        for request in requests_by_user[user_index]:
            prepared = prepare_request(sample, request)
            for batch in charged:
                batch.add(prepared)
        if is_progress_step(user_index + 1, max_users):
            logger.info("{}: prepared {}/{} users", condition, user_index + 1, max_users)
    return batches
