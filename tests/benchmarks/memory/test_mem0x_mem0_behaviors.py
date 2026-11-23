from __future__ import annotations

import json

import pytest

from benchmarks.common.vector_types import VectorStoreConfig
from benchmarks.memory.data import ConversationSample, Turn
from benchmarks.memory.mem0.catalog_replay import load_facts_into_memory
from benchmarks.memory.mem0.fact_catalog import make_memory_fact
from benchmarks.memory.mem0.llm_cache import CachedMemoryLLM
from benchmarks.memory.mem0.memory_handles import close_memory
from benchmarks.memory.mem0.provider import Mem0Settings, create_mem0_memory

WORD_BUCKETS = 32
ALICE_ADOPTED_REX = [{"role": "user", "content": "Alice: I adopted a beagle named Rex in Lisbon."}]


def bag_of_words(text):
    vector = [0.0] * WORD_BUCKETS
    for word in "".join(letter if letter.isalnum() else " " for letter in text.lower()).split():
        vector[sum(word.encode("utf-8")) % WORD_BUCKETS] += 1.0
    length = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / length for value in vector]


class BagOfWordsEmbedder:
    def embed(self, text, memory_action=None):
        return bag_of_words(text)

    def embed_batch(self, texts, memory_action="add"):
        return [bag_of_words(text) for text in texts]


class TwoFactLlm:
    def __init__(self, fact_ids):
        self.fact_ids = fact_ids
        self.prompts = []

    def generate_response(self, messages, **kwargs):
        user_prompt = messages[-1]["content"]
        self.prompts.append(user_prompt)
        said = prompt_section(user_prompt, "New Messages")
        facts = [
            {"id": fact_id, "text": f"{ordinal} fact about {said}", "attributed_to": "user"}
            for fact_id, ordinal in zip(self.fact_ids, ["First", "Second"])
        ]
        return json.dumps({"memory": facts})


def prompt_section(prompt, title):
    return prompt.split(f"## {title}\n", 1)[1].split("\n\n## ", 1)[0].strip()


@pytest.fixture
def new_mem0(monkeypatch, tmp_path):
    pytest.importorskip("en_core_web_sm")
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0-home"))
    monkeypatch.setenv("MEM0_TELEMETRY", "false")
    open_memories = []

    def create(store_name, fact_ids=("0", "1")):
        memory = create_mem0_memory(
            Mem0Settings(
                store_root=tmp_path / store_name,
                vector_config=VectorStoreConfig(backend="qdrant"),
                embedding_model="text-embedding-3-small",
                embedding_api_key="test-key",
                embedding_base_url=None,
                memory_llm_model="Qwen/Qwen2.5-7B-Instruct",
            )
        )
        memory.llm = TwoFactLlm(fact_ids)
        memory.embedding_model = BagOfWordsEmbedder()
        open_memories.append(memory)
        return memory

    yield create
    for memory in open_memories:
        close_memory(memory)


def rex_fact():
    turn = Turn("sample-1", "session_1", 1, 0, "Alice", "I adopted Rex.", "2026-01-02")
    sample = ConversationSample("sample-1", [turn], [], {})
    return make_memory_fact("Alice adopted a beagle named Rex in Lisbon.", sample, turn)


def linked_entities(memory):
    return {
        entity.payload["data"]: entity.payload["linked_memory_ids"]
        for entity in memory.entity_store.list(filters={"user_id": "sample-1"})
    }


def test_mem0_links_entities_when_it_infers_but_not_when_it_adds_a_fact_as_is(new_mem0):
    inferring = new_mem0("inferring")
    inferring.add(ALICE_ADOPTED_REX, user_id="sample-1", infer=True)
    adding_as_is = new_mem0("adding-as-is")
    load_facts_into_memory(adding_as_is, (rex_fact(),), link_entities=False)

    assert {"Rex", "Lisbon"} <= set(linked_entities(inferring))
    assert adding_as_is._entity_store is None
    assert linked_entities(adding_as_is) == {}


def extraction_prompt(**dates):
    from mem0.memory.main import generate_additive_extraction_prompt

    return generate_additive_extraction_prompt(
        existing_memories=[], new_messages="user: Alice likes tea.", last_k_messages=[], **dates
    )


def cache_key(cache_root, prompt):
    cache = CachedMemoryLLM(
        TwoFactLlm(("0", "1")), cache_root, "vllm", "model", "write", mem0_version="2.0.11"
    )
    cache.generate_response([{"role": "user", "content": prompt}])
    [entry] = cache.cache_dir.glob("*.json")
    return entry.name


def test_extraction_prompt_of_mem0_is_keyed_by_its_observation_date_but_not_by_todays_date(
    tmp_path,
):
    observed = "2026-01-02T00:00:00+00:00"
    written_in_july = extraction_prompt(current_date="2026-07-10", timestamp=observed)
    written_in_august = extraction_prompt(current_date="2026-08-11", timestamp=observed)
    observed_later = extraction_prompt(
        current_date="2026-07-10", timestamp="2026-01-03T00:00:00+00:00"
    )

    assert written_in_july != written_in_august
    assert cache_key(tmp_path / "july", written_in_july) == cache_key(
        tmp_path / "august", written_in_august
    )
    assert cache_key(tmp_path / "july", written_in_july) != cache_key(
        tmp_path / "later", observed_later
    )
