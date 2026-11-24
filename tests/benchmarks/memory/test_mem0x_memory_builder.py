from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from memory_config import make_memory_config, memory_runtime

from benchmarks.common.embedding_cache import CachedEmbedder
from benchmarks.memory.data import ConversationSample, QuestionAnswer, Turn
from benchmarks.memory.mem0 import memory_builder
from benchmarks.memory.mem0.fact_catalog import FactCatalogStore
from benchmarks.memory.mem0.memory_builder import SampleMemoryBuilder
from benchmarks.memory.mem0.provider import create_mem0_memory

CREATE_SECONDS = 2.0
ADD_SECONDS = 3.0
FINALIZE_SECONDS_PER_STORE = 5.0
CATALOG_LOAD_SECONDS = 100.0
QUESTION = "Where did Bob move?"
VALID_EXTRACTION = '{"memory":[{"id":"0","text":"anything","attributed_to":"user"}]}'


def conversation():
    return ConversationSample(
        sample_id="sample-1",
        turns=[
            Turn(
                "sample-1", "session_1", 1, 0, "Alice", "I adopted Rex.", "1:56 pm on 8 May, 2023"
            ),
            Turn(
                "sample-1", "session_1", 1, 1, "Bob", "I moved to Lisbon.", "1:56 pm on 8 May, 2023"
            ),
        ],
        qa=[QuestionAnswer("sample-1", "q0", QUESTION, "Lisbon", "4")],
        raw={"conversation": {"speaker_a": "Alice", "speaker_b": "Bob"}},
    )


class Clock:
    def __init__(self):
        self.now = 1000.0

    def advance(self, seconds):
        self.now += seconds


class JournalingStore:
    def __init__(self, name, backend, journal, clock):
        self.name = name
        self.config = SimpleNamespace(backend=backend)
        self.journal = journal
        self.clock = clock

    def reset(self):
        self.journal.append(f"{self.name} reset")

    def finalize(self):
        self.clock.advance(FINALIZE_SECONDS_PER_STORE)
        self.journal.append(f"{self.name} finalize")

    def close(self):
        self.journal.append(f"{self.name} close")

    def memory_stats(self):
        return {"jasper_vector_count": 2}


class JournalingMem0:
    def __init__(self, settings, journal, clock, *, extraction_response, add_failure):
        self.settings = settings
        self.journal = journal
        self.clock = clock
        self.add_failure = add_failure
        self.llm_calls = []
        self.llm = SimpleNamespace(
            generate_response=lambda messages, **kwargs: (
                self.llm_calls.append(kwargs) or extraction_response
            )
        )
        self.embedded = []
        self.embedding_model = SimpleNamespace(
            embed=lambda text, purpose=None: self.embedded.append((text, purpose)) or [1.0, 0.0]
        )
        self.vector_store = JournalingStore(
            "vector store", settings.vector_config.backend, journal, clock
        )
        self._entity_store = None

    @property
    def entity_store(self):
        self._entity_store = JournalingStore("entity store", "qdrant", self.journal, self.clock)
        return self._entity_store

    def add(self, messages, **options):
        self.journal.append(("add", messages[0]["content"], options["infer"]))
        self.clock.advance(ADD_SECONDS)
        if self.add_failure is not None:
            raise self.add_failure
        if options["infer"]:
            self.llm.generate_response(messages, response_format={"type": "json_object"})
            return {"results": [{"id": "mem0-id", "memory": f"Fact from {messages[0]['content']}"}]}
        return {"results": [{"id": options["metadata"]["fact_id"]}]}

    def _link_entities_for_memory(self, memory_id, text, filters):
        self.journal.append(("link entities", text))


class Mem0Factory:
    def __init__(self, monkeypatch, *, extraction_response=VALID_EXTRACTION, add_failure=None):
        self.journal = []
        self.clock = Clock()
        self.memories = []
        self.extraction_response = extraction_response
        self.add_failure = add_failure
        monkeypatch.setattr(memory_builder, "create_mem0_memory", self.create)
        monkeypatch.setattr(memory_builder.time, "perf_counter", lambda: self.clock.now)

    def create(self, settings):
        self.clock.advance(CREATE_SECONDS)
        self.journal.append("memory created")
        memory = JournalingMem0(
            settings,
            self.journal,
            self.clock,
            extraction_response=self.extraction_response,
            add_failure=self.add_failure,
        )
        self.memories.append(memory)
        return memory


def run_config(tmp_path, **values):
    settings = {
        "results_dir": tmp_path / "results",
        "run_id": "run",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "vector_backend": "qdrant",
        "embedding_cache_enabled": False,
    }
    return make_memory_config(
        **{**settings, **values},
        runtime=memory_runtime(
            storage={
                "cache_root": str(tmp_path / "cache"),
                "local_store_dir": str(tmp_path / "local"),
            }
        ),
    )


def extract_catalog(config):
    return SampleMemoryBuilder(config, memory_llm_cache_mode="write").build_memory(conversation())


def test_extraction_times_creation_and_building_and_counts_the_cache_traffic(monkeypatch, tmp_path):
    Mem0Factory(monkeypatch)
    config = run_config(tmp_path)

    first = extract_catalog(config)
    second = extract_catalog(config)

    assert list(first.metrics.items()) == [
        ("vector_backend", "qdrant"),
        ("memory_create_time_ms", CREATE_SECONDS * 1000),
        ("embedding_memory_build_time_ms", 2 * ADD_SECONDS * 1000),
        ("vector_index_build_time_ms", None),
        ("memory_setup_time_ms", (CREATE_SECONDS + 2 * ADD_SECONDS) * 1000),
        ("memory_input_turn_count", 2),
        ("memory_inferred_record_count", 2),
        ("memory_fact_catalog_loaded", 0),
        ("memory_llm_cache_hits", 0),
        ("memory_llm_cache_misses", 2),
    ]
    assert (second.metrics["memory_llm_cache_hits"], second.metrics["memory_llm_cache_misses"]) == (
        2,
        0,
    )


def test_replay_times_creation_loading_and_indexing_but_not_reading_the_catalog(
    monkeypatch, tmp_path
):
    mem0 = Mem0Factory(monkeypatch)
    config = run_config(tmp_path)
    extract_catalog(config)
    load_catalog = FactCatalogStore.load

    def slow_load(store, sample):
        mem0.clock.advance(CATALOG_LOAD_SECONDS)
        return load_catalog(store, sample)

    monkeypatch.setattr(FactCatalogStore, "load", slow_load)

    built = SampleMemoryBuilder(config).build_memory(conversation())

    assert list(built.metrics.items()) == [
        ("vector_backend", "qdrant"),
        ("memory_create_time_ms", CREATE_SECONDS * 1000),
        ("embedding_memory_build_time_ms", 2 * ADD_SECONDS * 1000),
        ("vector_index_build_time_ms", 2 * FINALIZE_SECONDS_PER_STORE * 1000),
        (
            "memory_setup_time_ms",
            (CREATE_SECONDS + 2 * ADD_SECONDS + 2 * FINALIZE_SECONDS_PER_STORE) * 1000,
        ),
        ("memory_input_turn_count", 2),
        ("memory_inferred_record_count", 2),
        ("memory_fact_catalog_loaded", 1),
        ("jasper_vector_count", 2),
    ]


WORD_BUCKETS = 32
UNTIDY_QUOTE = "Quote:   {said}\n"


def bag_of_words(text):
    vector = [0.0] * WORD_BUCKETS
    for word in "".join(letter if letter.isalnum() else " " for letter in text.lower()).split():
        vector[sum(word.encode("utf-8")) % WORD_BUCKETS] += 1.0
    length = sum(value * value for value in vector) ** 0.5 or 1.0
    return [value / length for value in vector]


class BagOfWordsEmbedder:
    def __init__(self):
        self.embedded = []

    def embed(self, text, memory_action=None):
        self.embedded.append((text, memory_action))
        return bag_of_words(text)

    def embed_batch(self, texts, memory_action="add"):
        return [self.embed(text, memory_action) for text in texts]


class TurnQuotingLlm:
    def __init__(self, quote):
        self.quote = quote
        self.prompts = []

    def generate_response(self, messages, **kwargs):
        user_prompt = messages[-1]["content"]
        self.prompts.append(user_prompt)
        said = user_prompt.split("## New Messages\n", 1)[1].split("\n\n## ", 1)[0].strip()
        quoted = {"id": "7", "text": self.quote.format(said=said), "attributed_to": "user"}
        return json.dumps({"memory": [quoted]})


class RealMem0WithScriptedModels:
    def __init__(self, monkeypatch, tmp_path, quote=UNTIDY_QUOTE):
        pytest.importorskip("en_core_web_sm")
        monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0-home"))
        monkeypatch.setenv("MEM0_TELEMETRY", "false")
        self.quote = quote
        self.llms = []
        self.embedders = []
        monkeypatch.setattr(memory_builder, "create_mem0_memory", self.create)

    def create(self, settings):
        memory = create_mem0_memory(settings)
        memory.llm = TurnQuotingLlm(self.quote)
        memory.embedding_model = BagOfWordsEmbedder()
        self.llms.append(memory.llm)
        self.embedders.append(memory.embedding_model)
        return memory


def preembed(config, sample):
    built = SampleMemoryBuilder(config, embedding_cache_mode="write").build_memory(
        sample, finalize_index=False
    )
    try:
        return built.facts, built.metrics
    finally:
        built.close()


@pytest.mark.parametrize("vector_backend", ["qdrant", "jasper"])
def test_catalog_extracted_by_mem0_replays_from_the_cache_and_embeds_only_the_query_live(
    monkeypatch, tmp_path, local_embedder, vector_backend
):
    mem0 = RealMem0WithScriptedModels(monkeypatch, tmp_path)
    config = run_config(tmp_path, vector_backend=vector_backend, embedding_cache_enabled=True)
    sample = conversation()

    facts, _ = preembed(config, sample)
    retriever, metrics = SampleMemoryBuilder(config).build_retriever_with_metrics(
        sample, finalize_index=vector_backend == "qdrant"
    )
    try:
        hits, retrieval = retriever.search(QUESTION, top_k=1)
        stored_ids = sorted(hit.id for hit in retriever.memory.vector_store.list())
    finally:
        retriever.close()

    assert [fact.text for fact in facts] == [
        "Quote: user: Alice: I adopted Rex.",
        "Quote: assistant: Bob: I moved to Lisbon.",
    ]
    assert stored_ids == sorted(fact.id for fact in facts)
    assert [hit.id for hit in hits] == [facts[1].id]
    assert hits[0].payload["memory"] == "Quote: assistant: Bob: I moved to Lisbon."
    assert hits[0].payload["source_turn_id"] == "sample-1:session_1:1"
    assert (metrics["vector_backend"], retrieval.vector_backend) == (vector_backend,) * 2
    assert metrics["memory_fact_catalog_loaded"] == 1
    assert retriever.built.embedding_cache_stats()["misses"] == 0
    assert mem0.llms[1].prompts == []
    embedded_live = mem0.embedders[1].embedded
    assert embedded_live[0] == (QUESTION, "search")
    assert {purpose for _, purpose in embedded_live} == {"search"}


def test_final_fact_text_that_mem0_stored_but_never_embedded_is_embedded_for_replay(
    monkeypatch, tmp_path, local_embedder
):
    mem0 = RealMem0WithScriptedModels(monkeypatch, tmp_path, quote=UNTIDY_QUOTE)
    config = run_config(tmp_path, embedding_cache_enabled=True)

    facts, _ = preembed(config, conversation())

    embedded_as_documents = [
        text for text, purpose in mem0.embedders[0].embedded if purpose == "add"
    ]
    text_mem0_embedded = "Quote:   user: Alice: I adopted Rex.\n"
    final_text = "Quote: user: Alice: I adopted Rex."
    assert facts[0].text == final_text
    assert embedded_as_documents.count(text_mem0_embedded) == 1
    assert embedded_as_documents.count(final_text) == 1
    read_only_cache = CachedEmbedder(
        BagOfWordsEmbedder(),
        cache_dir=config.embedding_cache_dir,
        model=config.embedding_model,
        mode="read",
    )
    assert read_only_cache.embed(final_text, "add") == pytest.approx(bag_of_words(final_text))


def test_repeated_extraction_by_mem0_sees_no_messages_of_the_earlier_one(
    monkeypatch, tmp_path, local_embedder
):
    mem0 = RealMem0WithScriptedModels(monkeypatch, tmp_path)
    config = run_config(tmp_path, embedding_cache_enabled=True)
    sample = conversation()

    first_facts, first_metrics = preembed(config, sample)
    second_facts, second_metrics = preembed(config, sample)

    first_llm, second_llm = mem0.llms
    assert len(first_llm.prompts) == 2
    assert "## Observation Date\n2023-05-08T13:56:00+00:00" in first_llm.prompts[0]
    assert "Alice: I adopted Rex." not in first_llm.prompts[0].split("## New Messages")[0]
    assert "Alice: I adopted Rex." in first_llm.prompts[1].split("## New Messages")[0]
    assert second_llm.prompts == []
    assert second_facts == first_facts
    assert (first_metrics["memory_llm_cache_hits"], first_metrics["memory_llm_cache_misses"]) == (
        0,
        2,
    )
    assert (second_metrics["memory_llm_cache_hits"], second_metrics["memory_llm_cache_misses"]) == (
        2,
        0,
    )
