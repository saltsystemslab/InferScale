from __future__ import annotations

import json
from typing import Any

from benchmarks.memory.protocol import (
    MEMORY_EXTRACTION_MAX_FACTS,
    MEMORY_EXTRACTION_MAX_TEXT_CHARS,
    MEMORY_EXTRACTION_MAX_TOKENS,
)

ATTRIBUTIONS = ("user", "assistant")
FACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "text": {"type": "string", "maxLength": MEMORY_EXTRACTION_MAX_TEXT_CHARS},
        "attributed_to": {"type": "string", "enum": list(ATTRIBUTIONS)},
        "linked_memory_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id", "text", "attributed_to"],
    "additionalProperties": False,
}
ALLOWED_FACT_FIELDS = frozenset(FACT_SCHEMA["properties"])
REQUIRED_FACT_FIELDS = frozenset(FACT_SCHEMA["required"])


class InvalidMemoryExtractionResponseError(RuntimeError):
    pass


def bounded_memory_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "mem0_memory_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "array",
                        "maxItems": MEMORY_EXTRACTION_MAX_FACTS,
                        "items": FACT_SCHEMA,
                    }
                },
                "required": ["memory"],
                "additionalProperties": False,
            },
        },
    }


def prepare_memory_extraction_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    effective_kwargs = dict(kwargs)
    if not _is_extraction_request(effective_kwargs):
        return effective_kwargs, False
    effective_kwargs["response_format"] = bounded_memory_response_format()
    effective_kwargs["max_tokens"] = MEMORY_EXTRACTION_MAX_TOKENS
    return effective_kwargs, True


def _is_extraction_request(kwargs: dict[str, Any]) -> bool:
    response_format = kwargs.get("response_format")
    return isinstance(response_format, dict) and response_format.get("type") == "json_object"


def validate_memory_extraction_response(response: Any) -> str:
    payload = _parse_payload(response)
    facts = payload["memory"]
    if not isinstance(facts, list):
        raise InvalidMemoryExtractionResponseError("Mem0 extraction 'memory' must be an array.")
    if len(facts) > MEMORY_EXTRACTION_MAX_FACTS:
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction returned {len(facts)} facts; "
            f"maximum is {MEMORY_EXTRACTION_MAX_FACTS}."
        )
    for position, fact in enumerate(facts):
        _validate_fact(position, fact)
    if _ids_are_sequential_as_prompted(facts):
        return response
    _renumber_unread_ids_by_position(facts)
    return json.dumps(payload, ensure_ascii=False)


def _parse_payload(response: Any) -> dict[str, Any]:
    if not isinstance(response, str):
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction returned {type(response).__name__}, expected a JSON string."
        )
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise InvalidMemoryExtractionResponseError(
            "Mem0 extraction returned malformed JSON "
            f"at character {exc.pos} of {len(response)}: {exc.msg}."
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"memory"}:
        raise InvalidMemoryExtractionResponseError(
            "Mem0 extraction JSON must contain exactly one top-level 'memory' field."
        )
    return payload


def _validate_fact(position: int, fact: Any) -> None:
    if not isinstance(fact, dict):
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction fact {position} must be an object."
        )
    if not REQUIRED_FACT_FIELDS.issubset(fact) or not set(fact).issubset(ALLOWED_FACT_FIELDS):
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction fact {position} has invalid or missing fields."
        )
    if not isinstance(fact["id"], str):
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction fact {position} id must be a string."
        )
    text = fact["text"]
    if not isinstance(text, str) or not text.strip():
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction fact {position} must have nonempty text."
        )
    if len(text) > MEMORY_EXTRACTION_MAX_TEXT_CHARS:
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction fact {position} has {len(text)} characters; "
            f"maximum is {MEMORY_EXTRACTION_MAX_TEXT_CHARS}."
        )
    if fact["attributed_to"] not in ATTRIBUTIONS:
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction fact {position} has invalid attributed_to."
        )
    if not _is_array_of_strings(fact.get("linked_memory_ids", [])):
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction fact {position} linked_memory_ids must be an array of strings."
        )


def _is_array_of_strings(candidate: Any) -> bool:
    return isinstance(candidate, list) and all(isinstance(entry, str) for entry in candidate)


def _ids_are_sequential_as_prompted(facts: list[dict[str, Any]]) -> bool:
    return all(fact["id"] == str(position) for position, fact in enumerate(facts))


def _renumber_unread_ids_by_position(facts: list[dict[str, Any]]) -> None:
    for position, fact in enumerate(facts):
        fact["id"] = str(position)
