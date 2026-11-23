from __future__ import annotations

import json

from benchmarks.memory.mem0.llm_protocol import (
    bounded_memory_response_format,
    validate_memory_extraction_response,
)
from benchmarks.memory.protocol import (
    MEMORY_EXTRACTION_MAX_FACTS,
    MEMORY_EXTRACTION_MAX_TEXT_CHARS,
)

BOUNDED_FORMAT = {
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
                            "attributed_to": {"type": "string", "enum": ["user", "assistant"]},
                            "linked_memory_ids": {"type": "array", "items": {"type": "string"}},
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


def fact(**changes):
    return {"id": "0", "text": "User likes blue.", "attributed_to": "user", **changes}


def extraction(*facts):
    return json.dumps({"memory": list(facts)})


def test_bounded_response_format_is_the_schema_sent_to_the_extraction_server():
    assert bounded_memory_response_format() == BOUNDED_FORMAT
    assert json.dumps(bounded_memory_response_format()) == json.dumps(BOUNDED_FORMAT)


def test_ids_that_guided_decoding_did_not_keep_sequential_are_renumbered_not_rejected():
    response = extraction(
        fact(id="1", text="first"),
        fact(id="1", text="second"),
        fact(id="e7b1c2d3", text="third", linked_memory_ids=["9"]),
    )

    renumbered = validate_memory_extraction_response(response)

    assert renumbered == extraction(
        fact(id="0", text="first"),
        fact(id="1", text="second"),
        fact(id="2", text="third", linked_memory_ids=["9"]),
    )
