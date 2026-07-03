from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..data import ConversationSample, QuestionAnswer
from ..prompts import RETRIEVAL_ANSWER_SYSTEM_PROMPT
from ..vector_types import SearchHit
from .context import format_memory_session
from .tokenization import encode_text_no_special

MEMORY_CONTEXT_INTRO = "The following is a conversation history between two people:\n\n"
MEMORY_SYSTEM_TEXT = RETRIEVAL_ANSWER_SYSTEM_PROMPT + "\n\n" + MEMORY_CONTEXT_INTRO
MemoryOrder = Literal[
    "retrieval",
    "session-index",
    "turn-index",
    "rank-zigzag",
    "retrieval-reversed",
]

_SENTINEL = "QZWXVUTSRPONMLK"
_MEMORY_FRAME_CACHE: dict[int, tuple[list[int], list[int]]] = {}


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
    if hit.payload.get("session_chunk_id"):
        return str(hit.payload["session_chunk_id"])
    if hit.payload.get("session_id"):
        sample_id = hit.payload.get("sample_id")
        session_id = str(hit.payload["session_id"])
        return f"{sample_id}:{session_id}" if sample_id else session_id
    return None


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

    sessions_by_id = {session.id: session for session in sample.sessions}
    # Older payloads may carry only the raw LoCoMo session_id. Accept them
    # within the current sample so stale vector stores fail less opaquely.
    sessions_by_id.update({session.session_id: session for session in sample.sessions})
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
        session_positions = {key: session.session_index for key, session in sessions_by_id.items()}
        return retrieval_session_ids, sorted(retrieval_session_ids, key=lambda session_id: session_positions[session_id])
    if memory_order == "rank-zigzag":
        return retrieval_session_ids, rank_zigzag_session_ids(retrieval_session_ids)
    if memory_order == "retrieval-reversed":
        return retrieval_session_ids, list(reversed(retrieval_session_ids))
    raise ValueError(f"Unsupported memory order: {memory_order!r}")


def rank_zigzag_session_ids(session_ids: list[str]) -> list[str]:
    front = session_ids[1::2]
    back = list(reversed(session_ids[::2]))
    return front + back


def memory_frame_token_ids(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Return system-role opening and closing tokens around memory content."""
    cache_key = id(tokenizer)
    cached = _MEMORY_FRAME_CACHE.get(cache_key)
    if cached is None:
        if _chat_template(tokenizer) is None:
            header = []
            bos_token_id = getattr(tokenizer, "bos_token_id", None)
            if bos_token_id is not None:
                header.append(bos_token_id)
            header.extend(encode_text_no_special(tokenizer, MEMORY_SYSTEM_TEXT))
            footer: list[int] = []
        else:
            full_ids = _templated_system_tokens(tokenizer, MEMORY_SYSTEM_TEXT)
            with_sentinel = _templated_system_tokens(tokenizer, MEMORY_SYSTEM_TEXT + _SENTINEL)
            split_at = _common_prefix_len(full_ids, with_sentinel)
            header = full_ids[:split_at]
            footer = full_ids[split_at:]
            if not footer:
                raise RuntimeError(
                    "Chat-template split produced an empty memory footer; "
                    "the sentinel likely merged with the memory header text."
                )
        cached = (header, footer)
        _MEMORY_FRAME_CACHE[cache_key] = cached
    header, footer = cached
    return list(header), list(footer)


def memory_frame_prefix_token_ids(tokenizer: Any) -> list[int]:
    header, _ = memory_frame_token_ids(tokenizer)
    return header


def memory_frame_suffix_token_ids(tokenizer: Any) -> list[int]:
    _, footer = memory_frame_token_ids(tokenizer)
    return footer


def build_memory_prompt_token_ids(
    tokenizer: Any,
    sample: ConversationSample,
    hits: list[SearchHit],
    *,
    memory_order: MemoryOrder = "session-index",
) -> MemoryPromptTokens:
    retrieval_session_ids, session_ids = ordered_memory_session_ids(sample, hits, memory_order=memory_order)
    sessions_by_id = {session.id: session for session in sample.sessions}
    sessions_by_id.update({session.session_id: session for session in sample.sessions})
    header, footer = memory_frame_token_ids(tokenizer)
    token_ids = list(header)
    for session_id in session_ids:
        token_ids.extend(encode_text_no_special(tokenizer, format_memory_session(sessions_by_id[session_id])))
    token_ids.extend(footer)
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
    user_text = kv_user_message_text(sample, qa)
    return tokenize_messages(tokenizer, [{"role": "user", "content": user_text}])


def kv_user_message_text(_sample: ConversationSample, qa: QuestionAnswer) -> str:
    return (
        "Based on the conversation above, answer concisely.\n"
        f"Question: {qa.question}"
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


def _templated_system_tokens(tokenizer: Any, system_body: str) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_body}],
            tokenize=True,
            add_generation_prompt=False,
        )
    )


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    for index in range(limit):
        if a[index] != b[index]:
            return index
    return limit
