from __future__ import annotations

from benchmarks.memory import protocol


def test_categories_are_named_as_the_upstream_memory_benchmarks_scorer_names_them():
    assert protocol.CATEGORY_NAMES == {
        "1": "multi-hop",
        "2": "temporal",
        "3": "open-domain",
        "4": "single-hop",
        "5": "adversarial",
    }
    assert list(protocol.CATEGORY_NAMES) == ["1", "2", "3", "4", "5"]


def test_adversarial_category_keeps_its_name_but_is_not_among_the_scored_categories():
    assert protocol.CATEGORY_NAMES["5"] == "adversarial"
    assert sorted(protocol.SCORED_CATEGORIES) == ["1", "2", "3", "4"]
    assert type(protocol.SCORED_CATEGORIES) is frozenset


def test_protocol_stamp_records_every_protocol_setting_in_order():
    assert list(protocol.PROTOCOL_STAMP.items()) == [
        ("memory_unit", "mem0-fact"),
        ("mem0_infer", True),
        ("memory_benchmarks_repository", "mem0ai/memory-benchmarks"),
        ("memory_benchmarks_commit", "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"),
        ("mem0ai_version", "2.0.11"),
        ("memory_extraction_response_protocol", "bounded-json-schema-v1"),
        ("memory_extraction_max_model_len", 16384),
        ("memory_extraction_max_tokens", 4096),
        ("memory_extraction_max_facts", 5),
        ("memory_extraction_max_text_chars", 600),
        ("memory_ingestion_protocol", "locomo-mem0-v3-chunk1-session-date-bounded-json-v2"),
        ("answer_prompt_protocol", "reverse-ranked-memory-block-safe-v1"),
        ("judge_prompt_protocol", "memory-benchmarks-json-jscore-v1"),
        ("context_window_unit", "turns"),
        ("context_window_semantics", "previous-turn-facts-prefix-discard-v1"),
    ]
