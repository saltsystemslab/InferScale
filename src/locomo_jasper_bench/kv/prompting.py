from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data import ConversationSample, QuestionAnswer, Turn, format_turn_for_memory
from ..vector_types import SearchHit
from .tokenization import encode_text_no_special

MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant that remembers details from past conversations. "
    "Answer questions based on the conversation history provided. "
    "The following is a conversation history between two people:\n\n"
)
MEMORY_TEMPLATE_PLACEHOLDER = "<<<LOCOMO_JASPER_MEMORY_GOES_HERE>>>"


@dataclass(slots=True, frozen=True)
class MemoryPromptTokens:
    token_ids: list[int]
    selected_turn_ids: list[str]


@dataclass(slots=True, frozen=True)
class MemoryScaffoldTokens:
    header_token_ids: list[int]
    footer_token_ids: list[int]


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


def extract_memory_scaffold_token_ids(tokenizer: Any) -> MemoryScaffoldTokens:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        templated = apply_chat_template(
            [{"role": "system", "content": MEMORY_SYSTEM_PROMPT + MEMORY_TEMPLATE_PLACEHOLDER}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if MEMORY_TEMPLATE_PLACEHOLDER in templated:
            header_text, footer_text = templated.split(MEMORY_TEMPLATE_PLACEHOLDER, 1)
            header_token_ids = encode_text_no_special(tokenizer, header_text)
            footer_token_ids = encode_text_no_special(tokenizer, footer_text)
            if not header_token_ids or not footer_token_ids:
                raise RuntimeError(
                    f"Empty memory scaffold tokens: header={len(header_token_ids)} footer={len(footer_token_ids)}."
                )
            return MemoryScaffoldTokens(
                header_token_ids=header_token_ids,
                footer_token_ids=footer_token_ids,
            )

    header_token_ids = encode_text_no_special(tokenizer, MEMORY_SYSTEM_PROMPT)
    if not header_token_ids:
        raise RuntimeError("Memory system prompt tokenized to zero tokens.")
    return MemoryScaffoldTokens(header_token_ids=header_token_ids, footer_token_ids=[])


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

    scaffold = extract_memory_scaffold_token_ids(tokenizer)
    token_ids = list(scaffold.header_token_ids)
    for turn_id in turn_ids:
        token_ids.extend(encode_text_no_special(tokenizer, format_kv_memory_turn(turns_by_id[turn_id])))
    token_ids.extend(scaffold.footer_token_ids)
    return MemoryPromptTokens(token_ids=token_ids, selected_turn_ids=turn_ids)


def build_kv_query_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    qa: QuestionAnswer,
) -> list[int]:
    return tokenize_messages(tokenizer, build_kv_query_messages(sample, qa))


def build_kv_query_messages(sample: ConversationSample, qa: QuestionAnswer) -> list[dict[str, str]]:
    del sample
    return [
        {
            "role": "user",
            "content": user_message_for_kv_question(qa.question),
        },
    ]


def user_message_for_kv_question(question: str) -> str:
    return (
        "Based on the conversation above, answer concisely.\n"
        f"Question: {question}"
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
