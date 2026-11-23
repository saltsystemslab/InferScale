from __future__ import annotations

import pytest

from benchmarks.memory.mem0.llm_cache import CachedMemoryLLM
from benchmarks.memory.mem0.llm_protocol import bounded_memory_response_format
from benchmarks.memory.protocol import MEMORY_EXTRACTION_MAX_TOKENS

CACHE_FOLDER = "vllm/model/mem0-2.0.11/bounded-json-schema-v1/context-16384"
DEFAULT_ENDPOINT_FOLDER = f"{CACHE_FOLDER}/endpoint-f578047bcc76d86f"
REMEMBER_BLUE = [{"role": "user", "content": "Remember blue."}]
REMEMBER_BLUE_KEY = "c29b5596430bcf58488bd24d8539764819a0404185a9800d84249dad87806a8b"
REMEMBER_BLUE_EXTRACTION_KEY = "5f3c64152b73461f1a91b51c70bc291d9a2f4aed81f7d1416847db68ff4dfb9a"
JSON_OBJECT = {"type": "json_object"}
VALID_EXTRACTION = '{"memory":[{"id":"0","text":"User likes blue.","attributed_to":"user"}]}'


class ScriptedLlm:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.config = {"model": "provider-model"}

    def generate_response(self, messages, *args, **kwargs):
        self.calls.append((messages, args, kwargs))
        return self.responses[min(len(self.calls), len(self.responses)) - 1]


def llm_cache(cache_root, llm, mode="write", **settings):
    return CachedMemoryLLM(
        llm, cache_root, "vllm", "model", mode, mem0_version="2.0.11", **settings
    )


def entry_names(cache):
    return sorted(path.name for path in cache.cache_dir.glob("*.json"))


def entry_key(cache_root, messages, *args, cache_settings=None, **kwargs):
    cache = llm_cache(cache_root, ScriptedLlm("response"), **(cache_settings or {}))
    cache.generate_response(messages, *args, **kwargs)
    [entry_name] = entry_names(cache)
    return entry_name.removesuffix(".json")


def test_cache_folder_names_provider_model_mem0_protocol_context_and_endpoint(tmp_path):
    default_endpoint = llm_cache(str(tmp_path), ScriptedLlm("response"))
    local_endpoint = llm_cache(
        tmp_path, ScriptedLlm("response"), endpoint="HTTP://LOCALHOST:8000/v1/"
    )

    assert default_endpoint.cache_dir == tmp_path / DEFAULT_ENDPOINT_FOLDER
    assert local_endpoint.cache_dir == tmp_path / CACHE_FOLDER / "endpoint-cbff7260a780c934"
    assert local_endpoint.endpoint == "http://localhost:8000/v1"


def test_entry_is_stored_under_the_digest_of_the_request(tmp_path):
    cache = llm_cache(tmp_path, ScriptedLlm("response"))

    assert cache.generate_response(REMEMBER_BLUE) == "response"

    entry = cache.cache_dir / f"{REMEMBER_BLUE_KEY}.json"
    assert entry_names(cache) == [entry.name]
    assert entry.read_text(encoding="utf-8") == '{"response":"response","version":1}\n'


def test_text_outside_ascii_is_keyed_as_written_not_as_escapes(tmp_path):
    unicode_key = entry_key(tmp_path, [{"role": "user", "content": "Ünïcödé tea 🚀"}])

    assert unicode_key == "c4ccb3f33d967972d02e52d483954fb98cecd3a7e9f9af585c8aeb6586ccefb3"


@pytest.mark.parametrize("temperature", [0.0])
def test_zero_temperature_is_left_out_of_the_key(tmp_path, temperature):
    key = entry_key(tmp_path, REMEMBER_BLUE, cache_settings={"temperature": temperature})

    assert key == REMEMBER_BLUE_KEY


def dated_prompt(observed, today):
    return f"## Observation Date\n{observed}\n\n## Current Date\n{today}"


def test_prompt_written_on_another_day_has_the_same_key(tmp_path):
    july_tenth = [{"role": "user", "content": dated_prompt("2026-07-10", "2026-07-10")}]
    july_eleventh = [{"role": "user", "content": dated_prompt("2026-07-10", "2026-07-11")}]

    assert entry_key(tmp_path / "tenth", july_tenth) == entry_key(
        tmp_path / "eleventh", july_eleventh
    )
    assert entry_key(tmp_path / "again", july_tenth) == (
        "a0b22ece72bb0572365f7300f8b5326ff175186c0e3afeef998433043df1dc6b"
    )


def test_extraction_request_is_sent_and_keyed_with_the_bounded_response_format(tmp_path):
    llm = ScriptedLlm(VALID_EXTRACTION)
    cache = llm_cache(tmp_path, llm)

    response = cache.generate_response(REMEMBER_BLUE, response_format=JSON_OBJECT)

    assert response == VALID_EXTRACTION
    assert llm.calls == [
        (
            REMEMBER_BLUE,
            (),
            {
                "response_format": bounded_memory_response_format(),
                "max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
            },
        )
    ]
    assert entry_names(cache) == [f"{REMEMBER_BLUE_EXTRACTION_KEY}.json"]


def test_mem0_reads_provider_attributes_such_as_config_through_the_cache(tmp_path):
    llm = ScriptedLlm("response")
    cache = llm_cache(tmp_path, llm, "read")

    assert cache.config is llm.config
    assert cache.calls is llm.calls
    assert cache.mode == "read"
    with pytest.raises(AttributeError, match="'ScriptedLlm' object has no attribute 'api_key'"):
        _ = cache.api_key
