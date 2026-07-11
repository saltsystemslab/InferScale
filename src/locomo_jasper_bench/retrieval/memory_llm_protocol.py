from __future__ import annotations

import json
from typing import Any

from ..protocol import (
    MEMORY_EXTRACTION_MAX_FACTS,
    MEMORY_EXTRACTION_MAX_TEXT_CHARS,
    MEMORY_EXTRACTION_MAX_TOKENS,
)


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
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {
                                    "type": "string",
                                    "maxLength": MEMORY_EXTRACTION_MAX_TEXT_CHARS,
                                },
                                "attributed_to": {
                                    "type": "string",
                                    "enum": ["user", "assistant"],
                                },
                                "linked_memory_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["id", "text", "attributed_to"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["memory"],
                "additionalProperties": False,
            },
        },
    }


def prepare_memory_extraction_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    effective = dict(kwargs)
    response_format = effective.get("response_format")
    if not isinstance(response_format, dict) or response_format.get("type") != "json_object":
        return effective, False
    effective["response_format"] = bounded_memory_response_format()
    effective["max_tokens"] = MEMORY_EXTRACTION_MAX_TOKENS
    return effective, True


def validate_memory_extraction_response(response: Any) -> str:
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
    memories = payload["memory"]
    if not isinstance(memories, list):
        raise InvalidMemoryExtractionResponseError("Mem0 extraction 'memory' must be an array.")
    if len(memories) > MEMORY_EXTRACTION_MAX_FACTS:
        raise InvalidMemoryExtractionResponseError(
            f"Mem0 extraction returned {len(memories)} facts; maximum is {MEMORY_EXTRACTION_MAX_FACTS}."
        )
    allowed_fields = {"id", "text", "attributed_to", "linked_memory_ids"}
    required_fields = {"id", "text", "attributed_to"}
    for index, memory in enumerate(memories):
        if not isinstance(memory, dict):
            raise InvalidMemoryExtractionResponseError(
                f"Mem0 extraction fact {index} must be an object."
            )
        if not required_fields.issubset(memory) or not set(memory).issubset(allowed_fields):
            raise InvalidMemoryExtractionResponseError(
                f"Mem0 extraction fact {index} has invalid or missing fields."
            )
        if memory["id"] != str(index):
            raise InvalidMemoryExtractionResponseError(
                f"Mem0 extraction fact {index} must have sequential id {str(index)!r}."
            )
        text = memory["text"]
        if not isinstance(text, str) or not text.strip():
            raise InvalidMemoryExtractionResponseError(
                f"Mem0 extraction fact {index} must have nonempty text."
            )
        if len(text) > MEMORY_EXTRACTION_MAX_TEXT_CHARS:
            raise InvalidMemoryExtractionResponseError(
                f"Mem0 extraction fact {index} has {len(text)} characters; "
                f"maximum is {MEMORY_EXTRACTION_MAX_TEXT_CHARS}."
            )
        if memory["attributed_to"] not in {"user", "assistant"}:
            raise InvalidMemoryExtractionResponseError(
                f"Mem0 extraction fact {index} has invalid attributed_to."
            )
        linked = memory.get("linked_memory_ids", [])
        if not isinstance(linked, list) or any(not isinstance(value, str) for value in linked):
            raise InvalidMemoryExtractionResponseError(
                f"Mem0 extraction fact {index} linked_memory_ids must be an array of strings."
            )
    return response
