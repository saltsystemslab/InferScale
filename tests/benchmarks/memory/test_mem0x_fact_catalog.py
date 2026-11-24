from __future__ import annotations

import json

import pytest

from benchmarks.memory.data import ConversationSample, Turn, sample_fingerprint
from benchmarks.memory.facts import memory_fact_from_hit
from benchmarks.memory.mem0.fact_catalog import FactCatalogStore, make_memory_fact

TEA_FACT_ID = "c8b971a6-1a31-5ecb-ab2c-feabada9bd4c"
CATALOG_FOLDER = (
    "fact-catalogs/mem0-2.0.11/vllm/Qwen_Qwen2.5-7B-Instruct/bounded-json-schema-v1"
    "/context-16384/tokens-4096/facts-5/text-600/endpoint-cbff7260a780c934"
    "/Qwen_Qwen3-Embedding-0.6B"
)


def conversation():
    return ConversationSample(
        sample_id="sample-1",
        turns=[
            Turn(
                sample_id="sample-1",
                session_id="session_1",
                session_index=1,
                turn_index=0,
                speaker="Alice",
                text="I like tea.",
                timestamp="1:56 pm on 8 May, 2023",
            ),
            Turn(
                sample_id="sample-1",
                session_id="session_2",
                session_index=2,
                turn_index=3,
                speaker="Bob",
                text="I prefer coffee.",
                timestamp="2026-01-02",
            ),
        ],
        qa=[],
        raw={"conversation": {"speaker_a": "Alice", "speaker_b": "Bob"}},
    )


def tea_fact():
    sample = conversation()
    return make_memory_fact("Alice likes tea.", sample, sample.turns[0])


def catalog_store(root, **changes):
    settings = {
        "provider": "vllm",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "endpoint": "http://localhost:8000/v1",
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "mem0_version": "2.0.11",
    }
    return FactCatalogStore(root, **{**settings, **changes})


def rewrite_catalog(catalog_path, change):
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    change(catalog)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")


def test_fact_id_is_derived_from_sample_source_turn_and_text():
    sample = conversation()

    fact = make_memory_fact("Alice likes tea.", sample, sample.turns[0])

    assert fact.id == TEA_FACT_ID
    assert make_memory_fact("Ünïcödé tea 🚀", sample, sample.turns[0]).id == (
        "d79bd4f4-6676-5a14-8722-7dfa02e28806"
    )
    assert make_memory_fact("Alice likes tea.", sample, sample.turns[1]).id == (
        "cea1d432-088c-5d58-ab08-bfd3d17a7d21"
    )


def test_source_turn_join_reads_the_aliases_when_the_source_turn_fields_are_missing():
    fact = tea_fact()
    with_aliases_only = fact.to_search_hit(1)
    for fields in (with_aliases_only.payload, with_aliases_only.payload["metadata"]):
        for source_turn_field in ("source_session_id", "source_turn_index", "source_turn_id"):
            del fields[source_turn_field]

    joined = memory_fact_from_hit(with_aliases_only)

    assert joined == memory_fact_from_hit(fact.to_search_hit(1))
    assert (joined.source_session_id, joined.source_turn_index, joined.source_turn_id) == (
        "session_1",
        0,
        "sample-1:session_1:0",
    )


def test_catalog_folder_names_everything_that_shaped_the_extraction(tmp_path):
    sample = conversation()

    store = catalog_store(tmp_path)

    assert store.catalog_dir == tmp_path / CATALOG_FOLDER
    assert store.path_for(sample) == (
        tmp_path / CATALOG_FOLDER / f"sample-1-{sample_fingerprint(sample)}.json"
    )


def test_catalog_format_version_6_is_exactly_this_payload_and_these_fact_fields(tmp_path):
    catalog_path = catalog_store(tmp_path).write(conversation(), [tea_fact()])

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    assert catalog["version"] == 6
    assert sorted(catalog) == [
        "embedding_model",
        "endpoint",
        "facts",
        "ingestion_protocol",
        "mem0_version",
        "memory_extraction_max_facts",
        "memory_extraction_max_model_len",
        "memory_extraction_max_text_chars",
        "memory_extraction_max_tokens",
        "memory_extraction_response_protocol",
        "memory_llm_temperature",
        "model",
        "provider",
        "sample_fingerprint",
        "sample_id",
        "vector_distance",
        "version",
    ]
    assert sorted(catalog["facts"][0]) == [
        "created_at",
        "id",
        "role",
        "sample_id",
        "source_session_id",
        "source_session_index",
        "source_turn_id",
        "source_turn_index",
        "speaker",
        "text",
        "timestamp_epoch",
    ]


def test_catalog_that_is_not_utf8_asks_to_be_replaced(tmp_path):
    sample = conversation()
    store = catalog_store(tmp_path)
    store.write(sample, [tea_fact()]).write_bytes(b"\xff\xfe\x00")

    with pytest.raises(RuntimeError) as error:
        store.load(sample)

    assert str(error.value) == (
        f"Corrupt Mem0 fact catalog at {store.path_for(sample)}. "
        "Rerun the extract-facts stage to replace it."
    )
    assert isinstance(error.value.__cause__, UnicodeDecodeError)


def test_catalog_of_another_format_version_is_rejected_before_anything_else(tmp_path):
    sample = conversation()
    store = catalog_store(tmp_path)
    catalog_path = store.write(sample, [tea_fact()])

    def downgrade(catalog):
        catalog["version"] = 5
        catalog["model"] = "other"
        catalog["facts"] = "not a list"

    rewrite_catalog(catalog_path, downgrade)

    with pytest.raises(RuntimeError, match="identity mismatch for version .* expected=6 actual=5"):
        store.load(sample)


@pytest.mark.parametrize(
    "field,mistyped,reason",
    [
        ("id", 7, "fact.id must be a string, got int."),
        ("source_turn_index", True, "fact.source_turn_index must be an integer, got a boolean."),
    ],
)
def test_catalog_fact_with_a_mistyped_field_is_rejected(tmp_path, field, mistyped, reason):
    sample = conversation()
    store = catalog_store(tmp_path)
    catalog_path = store.write(sample, [tea_fact()])
    rewrite_catalog(catalog_path, lambda catalog: catalog["facts"][0].update({field: mistyped}))

    with pytest.raises(RuntimeError) as error:
        store.load(sample)

    assert str(error.value) == f"Mem0 fact catalog contains an invalid fact: {catalog_path}"
    assert str(error.value.__cause__) == reason
