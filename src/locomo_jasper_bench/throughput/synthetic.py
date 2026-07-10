from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

from ..kv.prompting import (
    extract_memory_scaffold_token_ids,
    strip_duplicate_query_bos,
    tokenize_messages,
)
from ..kv.tokenization import encode_text_no_special

QUERIES = (
    "Suggest three research collaborations I should pursue.",
    "What career advice would you give me?",
    "Recommend three books given my interests.",
    "What are some upcoming events I should attend?",
    "Summarize the key themes from our recent conversations.",
)

MEMORY_TEMPLATES = (
    "I work as a {role} at {company}. My main focus is on {topic}.",
    "Last week I mentioned that I am interested in {topic} and want to explore it more deeply.",
    "I have a meeting scheduled with {person} to discuss {topic} next {day}.",
    "My favorite approach to {topic} is using {method}, which I have been refining for {duration}.",
    "I recently read a paper about {topic} by {person} and found the section on {subtopic} particularly insightful.",
    "For my current project on {topic}, I need to benchmark against {baseline} and compare {metric} across {num} configurations.",
    "I prefer {preference_a} over {preference_b} for {domain}, especially in {context}.",
    "My team consists of {num} people working on {topic}. Key collaborators include {person} and {person2}.",
    "I attended {event} last month where {person} presented work on {topic}. The key takeaway concerned {subtopic}.",
    "My long-term research goal is to build {system} that can handle {scale} while maintaining {property}.",
)

FILLERS = {
    "role": ("software engineer", "data scientist", "researcher", "professor", "ML engineer"),
    "company": ("Anthropic", "Google", "Meta", "Microsoft", "a startup"),
    "topic": ("distributed systems", "machine learning", "NLP", "computer vision", "data structures"),
    "person": ("Dr. Smith", "Prof. Johnson", "Alice", "Bob", "Dr. Chen"),
    "person2": ("Dr. Lee", "Prof. Garcia", "Carlos", "Diana", "Dr. Kumar"),
    "day": ("Monday", "Wednesday", "Friday"),
    "method": ("gradient descent", "beam search", "dynamic programming", "randomized algorithms"),
    "duration": ("3 months", "a year", "6 weeks"),
    "subtopic": ("scalability", "optimization", "memory efficiency", "parallelism"),
    "baseline": ("the state of the art", "GPT-4", "a simple heuristic", "the previous version"),
    "metric": ("throughput", "latency", "accuracy", "F1 score"),
    "num": ("5", "10", "20", "50"),
    "preference_a": ("Python", "Rust", "C++", "Julia"),
    "preference_b": ("Java", "Go", "JavaScript", "MATLAB"),
    "domain": ("systems programming", "data analysis", "model training", "deployment"),
    "context": ("production environments", "research prototyping", "large-scale experiments"),
    "event": ("NeurIPS", "SIGMOD", "VLDB", "SC", "ICML"),
    "system": ("a real-time inference engine", "a distributed index", "a streaming pipeline"),
    "scale": ("billions of records", "millions of QPS", "petabytes of data"),
    "property": ("low latency", "high throughput", "strong consistency"),
}


@dataclass(frozen=True, slots=True)
class SyntheticMemory:
    user_id: str
    memory_token_ids: tuple[int, ...]
    body_token_ids: tuple[int, ...]
    entries: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SyntheticRequest:
    user_id: str
    user_index: int
    request_index: int
    query: str


def build_synthetic_memory(
    tokenizer: Any,
    *,
    user_index: int,
    memory_tokens: int,
    seed: int,
    entry_tokens: int = 128,
) -> SyntheticMemory:
    if entry_tokens <= 0:
        raise ValueError("entry_tokens must be greater than zero.")
    scaffold = extract_memory_scaffold_token_ids(tokenizer)
    body_target = memory_tokens - len(scaffold.header_token_ids) - len(scaffold.footer_token_ids)
    if body_target <= 0:
        raise ValueError(
            f"memory_tokens={memory_tokens} is too small for the model's "
            f"{len(scaffold.header_token_ids) + len(scaffold.footer_token_ids)}-token memory scaffold."
        )

    rng = random.Random(f"{seed}:memory:{memory_tokens}:{user_index}")
    body_ids: list[int] = []
    identity = (
        f"This memory belongs to benchmark user {user_index:04d}. "
        f"The stable account identifier is throughput-user-{user_index:04d}.\n"
    )
    body_ids.extend(encode_text_no_special(tokenizer, identity))
    while len(body_ids) < body_target:
        body_ids.extend(encode_text_no_special(tokenizer, _render_fact(rng) + "\n"))
    body_ids = body_ids[:body_target]

    memory_token_ids = [
        *scaffold.header_token_ids,
        *body_ids,
        *scaffold.footer_token_ids,
    ]
    if len(memory_token_ids) != memory_tokens:
        raise AssertionError("Synthetic memory construction did not produce the requested token count.")
    entries = tuple(
        text
        for start in range(0, len(body_ids), entry_tokens)
        if (text := _decode(tokenizer, body_ids[start : start + entry_tokens]).strip())
    )
    if not entries:
        raise RuntimeError("Synthetic memory decoded to no Mem0 entries.")
    return SyntheticMemory(
        user_id=user_id(user_index),
        memory_token_ids=tuple(memory_token_ids),
        body_token_ids=tuple(body_ids),
        entries=entries,
    )


def build_requests(*, num_users: int, requests_per_user: int, seed: int) -> list[SyntheticRequest]:
    requests: list[SyntheticRequest] = []
    for user_index in range(num_users):
        for request_index in range(requests_per_user):
            rng = random.Random(f"{seed}:query:{user_index}:{request_index}")
            requests.append(
                SyntheticRequest(
                    user_id=user_id(user_index),
                    user_index=user_index,
                    request_index=request_index,
                    query=rng.choice(QUERIES),
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


def build_memory_prompt(tokenizer: Any, memory_token_ids: Iterable[int], query: str) -> list[int]:
    memory_ids = list(memory_token_ids)
    query_ids = tokenize_messages(tokenizer, [{"role": "user", "content": query}])
    query_ids, _ = strip_duplicate_query_bos(
        tokenizer,
        memory_token_ids=memory_ids,
        query_token_ids=query_ids,
    )
    return memory_ids + query_ids


def build_retrieval_prompt(tokenizer: Any, memories: Iterable[str], query: str) -> list[int]:
    memory_lines = [f"- {memory.strip()}" for memory in memories if memory.strip()]
    if memory_lines:
        system_message = "You are a helpful assistant. Relevant memories:\n" + "\n".join(memory_lines)
    else:
        system_message = "You are a helpful assistant. Answer concisely."
    return tokenize_messages(
        tokenizer,
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ],
    )


def user_id(user_index: int) -> str:
    return f"user_{user_index:04d}"


def _render_fact(rng: random.Random) -> str:
    text = rng.choice(MEMORY_TEMPLATES)
    for key, values in FILLERS.items():
        placeholder = "{" + key + "}"
        while placeholder in text:
            text = text.replace(placeholder, rng.choice(values), 1)
    return text


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise RuntimeError("Tokenizer has no decode method.")
    try:
        return str(
            decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(decode(token_ids, skip_special_tokens=True))
