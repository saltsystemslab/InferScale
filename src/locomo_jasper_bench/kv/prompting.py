from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..data import ConversationSample, QuestionAnswer, SessionChunk, format_session_for_memory
from ..prompts import RETRIEVAL_ANSWER_SYSTEM_PROMPT
from ..vector_types import SearchHit
from .tokenization import encode_text_no_special

MEMORY_PREFIX_TEXT = "Retrieved memory context:\n"
MemoryOrder = Literal["retrieval", "session-index", "turn-index", "retrieval-reversed"]


@dataclass(slots=True, frozen=True)
class MemoryPromptTokens:
    token_ids: list[int]
    selected_session_ids: list[str]
    retrieval_session_ids: list[str]
    memory_order: str


@dataclass(slots=True, frozen=True)
class KvPromptTokens:
    memory_token_ids: list[int]
    query_token_ids: list[int]
    prompt_token_ids: list[int]
    stripped_query_bos: bool


def hit_session_id(hit: SearchHit) -> str | None:
    metadata = hit.payload.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("session_chunk_id"):
            return str(metadata["session_chunk_id"])
        if metadata.get("session_id"):
            sample_id = metadata.get("sample_id")
            session_id = str(metadata["session_id"])
            return f"{sample_id}:{session_id}" if sample_id else session_id
        if metadata.get("turn_id"):
            return session_id_from_turn_id(str(metadata["turn_id"]))
    if hit.payload.get("session_chunk_id"):
        return str(hit.payload["session_chunk_id"])
    if hit.payload.get("session_id"):
        sample_id = hit.payload.get("sample_id")
        session_id = str(hit.payload["session_id"])
        return f"{sample_id}:{session_id}" if sample_id else session_id
    if hit.payload.get("turn_id"):
        return session_id_from_turn_id(str(hit.payload["turn_id"]))
    return None


def session_id_from_turn_id(turn_id: str) -> str | None:
    base, separator, _turn_index = turn_id.rpartition(":")
    if not separator or not base:
        return None
    return base


def selected_session_ids(hits: list[SearchHit]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        session_id = hit_session_id(hit)
        if session_id and session_id not in seen:
            ids.append(session_id)
            seen.add(session_id)
    return ids


def ordered_memory_session_ids(
    sample: ConversationSample,
    hits: list[SearchHit],
    *,
    memory_order: MemoryOrder = "session-index",
) -> tuple[list[str], list[str]]:
    hit_session_ids = selected_session_ids(hits)
    if not hit_session_ids:
        raise RuntimeError("Cannot compose memory because retrieval returned no session ids.")

    sessions_by_id = sessions_by_lookup_id(sample)
    missing = [session_id for session_id in hit_session_ids if session_id not in sessions_by_id]
    if missing:
        raise RuntimeError("Retrieved memory sessions were not found in the sample: " + ", ".join(missing[:5]))

    retrieval_session_ids: list[str] = []
    seen: set[str] = set()
    for session_id in hit_session_ids:
        canonical_id = sessions_by_id[session_id].id
        if canonical_id not in seen:
            retrieval_session_ids.append(canonical_id)
            seen.add(canonical_id)

    if memory_order == "retrieval":
        return retrieval_session_ids, retrieval_session_ids
    if memory_order in {"session-index", "turn-index"}:
        session_positions = {
            lookup_id: session.session_index
            for lookup_id, session in sessions_by_id.items()
        }
        return retrieval_session_ids, sorted(retrieval_session_ids, key=lambda session_id: session_positions[session_id])
    if memory_order == "retrieval-reversed":
        return retrieval_session_ids, list(reversed(retrieval_session_ids))
    raise ValueError(f"Unsupported memory order: {memory_order!r}")


def sessions_by_lookup_id(sample: ConversationSample) -> dict[str, SessionChunk]:
    sessions_by_id = {session.id: session for session in sample.sessions}
    sessions_by_id.update({session.session_id: session for session in sample.sessions})
    return sessions_by_id


def format_kv_memory_session(session: SessionChunk) -> str:
    return format_session_for_memory(session).strip() + "\n\n"


def build_memory_prompt_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    hits: list[SearchHit],
    *,
    memory_order: MemoryOrder = "session-index",
) -> MemoryPromptTokens:
    retrieval_session_ids, session_ids = ordered_memory_session_ids(sample, hits, memory_order=memory_order)
    sessions_by_id = sessions_by_lookup_id(sample)
    token_ids = encode_text_no_special(tokenizer, MEMORY_PREFIX_TEXT)
    for session_id in session_ids:
        token_ids.extend(encode_text_no_special(tokenizer, format_kv_memory_session(sessions_by_id[session_id])))
    return MemoryPromptTokens(
        token_ids=token_ids,
        selected_session_ids=session_ids,
        retrieval_session_ids=retrieval_session_ids,
        memory_order=memory_order,
    )


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
