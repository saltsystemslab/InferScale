from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data import ConversationSample, QuestionAnswer, Turn, format_turn_for_memory
from ..prompts import RETRIEVAL_ANSWER_SYSTEM_PROMPT
from ..vector_types import SearchHit
from .tokenization import encode_text_no_special

MEMORY_PREFIX_TEXT = "Retrieved memory context:\n"


@dataclass(slots=True, frozen=True)
class MemoryPromptTokens:
    token_ids: list[int]
    selected_turn_ids: list[str]


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


def format_kv_memory_turn(turn: Turn) -> str:
    return format_turn_for_memory(turn).strip() + "\n"


def build_memory_prompt_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    hits: list[SearchHit],
) -> MemoryPromptTokens:
    turn_ids = selected_turn_ids(hits)
    if not turn_ids:
        raise RuntimeError("Cannot compose KV-equivalence prompt because retrieval returned no turn ids.")

    turns_by_id = {turn.id: turn for turn in sample.turns}
    missing = [turn_id for turn_id in turn_ids if turn_id not in turns_by_id]
    if missing:
        raise RuntimeError("Retrieved memory chunks were not found in the sample: " + ", ".join(missing[:5]))

    token_ids = encode_text_no_special(tokenizer, MEMORY_PREFIX_TEXT)
    for turn_id in turn_ids:
        token_ids.extend(encode_text_no_special(tokenizer, format_kv_memory_turn(turns_by_id[turn_id])))
    return MemoryPromptTokens(token_ids=token_ids, selected_turn_ids=turn_ids)


def build_kv_query_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    qa: QuestionAnswer,
) -> list[int]:
    return tokenize_messages(tokenizer, build_kv_query_messages(sample, qa))


def build_kv_query_messages(sample: ConversationSample, qa: QuestionAnswer) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                RETRIEVAL_ANSWER_SYSTEM_PROMPT
                + " Relevant retrieved memory is available before the current question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation id: {sample.sample_id}\n\n"
                f"Question: {qa.question}\n\n"
                "Answer:"
            ),
        },
    ]


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
