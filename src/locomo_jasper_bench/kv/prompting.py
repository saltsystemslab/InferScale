from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..data import ConversationSample, QuestionAnswer
from ..prompts import RETRIEVAL_ANSWER_SYSTEM_PROMPT
from ..vector_types import SearchHit
from .context import format_memory_turn
from .tokenization import encode_text_no_special

MEMORY_PREFIX_TEXT = "Retrieved memory context:\n"
MEMORY_SYSTEM_TEXT = (
    RETRIEVAL_ANSWER_SYSTEM_PROMPT
    + " Relevant retrieved memory is available before the current question.\n\n"
    + MEMORY_PREFIX_TEXT
)
MemoryOrder = Literal["retrieval", "turn-index", "rank-zigzag", "retrieval-reversed"]

# Two system bodies with no shared words, used to locate the chat template's
# scaffolding empirically: tokens common to both templated conversations are
# scaffolding, tokens that differ belong to the varying system body.
_PROBE_SYSTEM_BODY_A = "alpha oak river seven"
_PROBE_SYSTEM_BODY_B = "zeta glass mountain"
_PROBE_USER_TEXT = "probe question"

_MEMORY_FRAME_CACHE: dict[int, list[int]] = {}


@dataclass(slots=True, frozen=True)
class MemoryPromptTokens:
    token_ids: list[int]
    selected_turn_ids: list[str]
    retrieval_turn_ids: list[str]
    memory_order: str


@dataclass(slots=True, frozen=True)
class KvPromptTokens:
    memory_token_ids: list[int]
    query_token_ids: list[int]
    prompt_token_ids: list[int]
    stripped_query_bos: bool


def hit_turn_id(hit: SearchHit) -> str | None:
    metadata = hit.payload.get("metadata")
    if isinstance(metadata, dict) and metadata.get("turn_id"):
        return str(metadata["turn_id"])
    if hit.payload.get("turn_id"):
        return str(hit.payload["turn_id"])
    return None


def selected_turn_ids(hits: list[SearchHit]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        turn_id = hit_turn_id(hit)
        if turn_id and turn_id not in seen:
            ids.append(turn_id)
            seen.add(turn_id)
    return ids


def ordered_memory_turn_ids(
    sample: ConversationSample,
    hits: list[SearchHit],
    *,
    memory_order: MemoryOrder = "retrieval",
) -> tuple[list[str], list[str]]:
    retrieval_turn_ids = selected_turn_ids(hits)
    if not retrieval_turn_ids:
        raise RuntimeError("Cannot compose memory because retrieval returned no turn ids.")

    turns_by_id = {turn.id: turn for turn in sample.turns}
    missing = [turn_id for turn_id in retrieval_turn_ids if turn_id not in turns_by_id]
    if missing:
        raise RuntimeError("Retrieved memory chunks were not found in the sample: " + ", ".join(missing[:5]))

    if memory_order == "retrieval":
        return retrieval_turn_ids, retrieval_turn_ids
    if memory_order == "turn-index":
        turn_positions = {turn.id: index for index, turn in enumerate(sample.turns)}
        return retrieval_turn_ids, sorted(retrieval_turn_ids, key=lambda turn_id: turn_positions[turn_id])
    if memory_order == "rank-zigzag":
        return retrieval_turn_ids, rank_zigzag_turn_ids(retrieval_turn_ids)
    if memory_order == "retrieval-reversed":
        return retrieval_turn_ids, list(reversed(retrieval_turn_ids))
    raise ValueError(f"Unsupported memory order: {memory_order!r}")


def rank_zigzag_turn_ids(turn_ids: list[str]) -> list[str]:
    front = turn_ids[1::2]
    back = list(reversed(turn_ids[::2]))
    return front + back


def memory_frame_prefix_token_ids(tokenizer: Any) -> list[int]:
    """Chat-template tokens that precede the system message content (BOS, headers).

    The memory block is injected at positions [0, N), so it must carry the
    scaffolding that normally opens a prompt. Derived empirically as the
    longest common prefix of two templated conversations that differ only in
    the system body; assumes the template renders the system message first.
    Without a chat template, falls back to a bare BOS.
    """
    cache_key = id(tokenizer)
    cached = _MEMORY_FRAME_CACHE.get(cache_key)
    if cached is None:
        if _chat_template(tokenizer) is None:
            bos_token_id = getattr(tokenizer, "bos_token_id", None)
            cached = [bos_token_id] if bos_token_id is not None else []
        else:
            tokens_a = _templated_tokens(tokenizer, _PROBE_SYSTEM_BODY_A, _PROBE_USER_TEXT)
            tokens_b = _templated_tokens(tokenizer, _PROBE_SYSTEM_BODY_B, _PROBE_USER_TEXT)
            cached = tokens_a[: _common_prefix_len(tokens_a, tokens_b)]
        _MEMORY_FRAME_CACHE[cache_key] = cached
    return list(cached)


def build_memory_prompt_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    hits: list[SearchHit],
    *,
    memory_order: MemoryOrder = "retrieval",
) -> MemoryPromptTokens:
    retrieval_turn_ids, turn_ids = ordered_memory_turn_ids(sample, hits, memory_order=memory_order)
    turns_by_id = {turn.id: turn for turn in sample.turns}
    token_ids = memory_frame_prefix_token_ids(tokenizer)
    token_ids.extend(encode_text_no_special(tokenizer, MEMORY_SYSTEM_TEXT))
    for turn_id in turn_ids:
        token_ids.extend(encode_text_no_special(tokenizer, format_memory_turn(turns_by_id[turn_id])))
    return MemoryPromptTokens(
        token_ids=token_ids,
        selected_turn_ids=turn_ids,
        retrieval_turn_ids=retrieval_turn_ids,
        memory_order=memory_order,
    )


def build_kv_query_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    qa: QuestionAnswer,
) -> list[int]:
    """Tokens that follow the memory block: system-role close, user turn, generation prompt.

    Derived as the longest common suffix of two templated conversations that
    differ only in the system body, so `memory + query` concatenates to exactly
    the sequence the chat template produces for (system=memory, user=question).
    """
    user_text = kv_user_message_text(sample, qa)
    if _chat_template(tokenizer) is None:
        return tokenize_messages(tokenizer, [{"role": "user", "content": user_text}])
    tokens_a = _templated_tokens(tokenizer, _PROBE_SYSTEM_BODY_A, user_text)
    tokens_b = _templated_tokens(tokenizer, _PROBE_SYSTEM_BODY_B, user_text)
    return tokens_a[len(tokens_a) - _common_suffix_len(tokens_a, tokens_b):]


def kv_user_message_text(sample: ConversationSample, qa: QuestionAnswer) -> str:
    return (
        f"Conversation id: {sample.sample_id}\n\n"
        f"Question: {qa.question}\n\n"
        "Answer:"
    )


def build_kv_equivalence_prompt_token_ids(
    tokenizer: Any,
    memory_token_ids: list[int],
    sample: ConversationSample,
    qa: QuestionAnswer,
) -> KvPromptTokens:
    query_token_ids = build_kv_query_token_ids(tokenizer, sample, qa)
    query_token_ids, stripped_query_bos = strip_duplicate_query_bos(
        tokenizer,
        memory_token_ids=memory_token_ids,
        query_token_ids=query_token_ids,
    )
    return KvPromptTokens(
        memory_token_ids=list(memory_token_ids),
        query_token_ids=query_token_ids,
        prompt_token_ids=list(memory_token_ids) + query_token_ids,
        stripped_query_bos=stripped_query_bos,
    )


def strip_duplicate_query_bos(
    tokenizer: Any,
    *,
    memory_token_ids: list[int],
    query_token_ids: list[int],
) -> tuple[list[int], bool]:
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if (
        bos_token_id is not None
        and memory_token_ids
        and query_token_ids
        and memory_token_ids[0] == bos_token_id
        and query_token_ids[0] == bos_token_id
    ):
        return list(query_token_ids[1:]), True
    return list(query_token_ids), False


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return list(
            apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

    text = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("Tokenizer has neither apply_chat_template nor encode.")
    return list(encode(text))


def _chat_template(tokenizer: Any) -> Any:
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        return None
    return getattr(tokenizer, "chat_template", None)


def _templated_tokens(tokenizer: Any, system_body: str, user_text: str) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_body},
                {"role": "user", "content": user_text},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for index in range(limit):
        if a[index] != b[index]:
            return index
    return limit


def _common_suffix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for offset in range(1, limit + 1):
        if a[-offset] != b[-offset]:
            return offset - 1
    return limit
