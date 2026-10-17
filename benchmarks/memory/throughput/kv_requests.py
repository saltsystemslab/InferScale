"""Per-request retrieval, KV composition, registration, and verification for kv_injection.

Every request is prepared before vLLM starts. The Jasper search keeps the
retrieved IDs on the GPU, the chunk store gathers those rows into the request's
memory, and the composed KV is registered in the connector namespace under the
request's memory user id. The prompt-injection token stream for the same facts
is built afterwards only to verify the composed tokens; that check and the
registration are timed separately and excluded from QPS.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

from benchmarks.memory.data import ConversationSample
from benchmarks.memory.mem0.fact_catalog import MemoryFact
from benchmarks.memory.prompting import build_memory_prompt_token_ids
from benchmarks.memory.throughput.config import ThroughputConfig
from benchmarks.memory.throughput.kv_corpus import PrecomputedCorpus
from benchmarks.memory.throughput.stores import (
    SampleStores,
    StoreSearch,
    materialize_store_device_result,
    search_store_for_kv,
    store_stable_id_items,
)
from benchmarks.memory.throughput.workload import (
    LocomoRequest,
    is_progress_step,
    request_prompt_token_ids,
    sample_for_user,
)
from inferscale.v1.kv.chunk_store import (
    build_device_chunk_row_map,
    fetch_device_chunk,
    fetch_device_chunks,
    release_chunks,
)
from inferscale.v1.kv.prompt import ScaffoldTokens, require_identical_token_ids
from inferscale.v1.kv.registry import register_user_memory
from inferscale.v1.types import DeviceSearchResult, KVChunk


@dataclass(slots=True)
class RequestTimings:
    """Seconds accumulated over every prepared request, split by phase."""

    retrieval_s: float = 0.0
    vector_search_s: float = 0.0
    compose_s: float = 0.0
    store_write_s: float = 0.0
    verify_s: float = 0.0
    prompt_build_s: float = 0.0


@dataclass(slots=True)
class ComposedRequests:
    """Every request's prompt and memory routing, ready for the measured batch."""

    prompts: list[dict[str, list[int]]] = field(default_factory=list)
    memory_user_ids: list[str] = field(default_factory=list)
    first_memory_user_id: str | None = None
    first_memory_token_ids: list[int] | None = None
    timings: RequestTimings = field(default_factory=RequestTimings)
    fact_count_total: int = 0

    def add(
        self, memory_user_id: str, prompt_token_ids: list[int], memory_token_ids: list[int]
    ) -> None:
        if self.first_memory_user_id is None:
            self.first_memory_user_id = memory_user_id
            self.first_memory_token_ids = list(memory_token_ids)
        self.prompts.append({"prompt_token_ids": prompt_token_ids})
        self.memory_user_ids.append(memory_user_id)


@dataclass(slots=True)
class SampleRetrieval:
    """A sample's Jasper store and the GPU map from its stable IDs to chunk store rows."""

    memory: Any
    row_map: Any


@dataclass(slots=True)
class RequestContext:
    """What composing a request needs, shared by every request of the run."""

    config: ThroughputConfig
    corpus: PrecomputedCorpus
    retrieval: dict[str, SampleRetrieval]
    namespace: str
    scaffold: ScaffoldTokens


def build_sample_retrieval(
    corpus: PrecomputedCorpus, stores: SampleStores
) -> tuple[dict[str, SampleRetrieval], float]:
    """Pair each sample's Jasper store with the device row map over its chunk store."""
    started = time.perf_counter()
    retrieval = {}
    for sample_id, memory in stores.memories.items():
        row_map = build_device_chunk_row_map(
            corpus.chunk_store(sample_id), store_stable_id_items(memory)
        )
        retrieval[sample_id] = SampleRetrieval(memory=memory, row_map=row_map)
    return retrieval, time.perf_counter() - started


def compose_requests(
    context: RequestContext,
    requests_by_user: Mapping[int, Sequence[LocomoRequest]],
    samples: Sequence[ConversationSample],
    facts_by_sample: Mapping[str, tuple[MemoryFact, ...]],
) -> ComposedRequests:
    """Prepare every user's requests in order; memory user ids follow request order."""
    composed = ComposedRequests()
    num_users = len(requests_by_user)
    for user_index in range(num_users):
        sample = sample_for_user(samples, user_index)
        composed.fact_count_total += len(facts_by_sample[sample.sample_id])
        for request in requests_by_user[user_index]:
            memory_user_id = f"request-{len(composed.prompts):05d}"
            prompt_token_ids, memory_token_ids = compose_request(
                context, sample, request, memory_user_id, composed.timings
            )
            composed.add(memory_user_id, prompt_token_ids, memory_token_ids)
        if is_progress_step(user_index + 1, num_users):
            logger.info("kv_injection: retrieved {}/{} users", user_index + 1, num_users)
    return composed


def compose_request(
    context: RequestContext,
    sample: ConversationSample,
    request: LocomoRequest,
    memory_user_id: str,
    timings: RequestTimings,
) -> tuple[list[int], list[int]]:
    """Retrieve, compose, register, and verify one request; returns its prompt and memory tokens."""
    retrieval = context.retrieval[sample.sample_id]
    search = _search(retrieval.memory, request.query, context.config.top_k, timings)
    kv_by_layer, memory_token_ids = _compose_memory(context, sample, search.result, timings)
    _register(context.namespace, memory_user_id, kv_by_layer, memory_token_ids, timings)
    _verify(context, retrieval.memory, sample, search.result, memory_token_ids, timings)
    started = time.perf_counter()
    prompt_token_ids = request_prompt_token_ids(context.corpus.tokenizer, memory_token_ids, request)
    timings.prompt_build_s += time.perf_counter() - started
    return prompt_token_ids, memory_token_ids


def synchronize_cuda() -> None:
    torch.cuda.synchronize()


def _search(
    memory: Any, query: str, top_k: int, timings: RequestTimings
) -> StoreSearch[DeviceSearchResult]:
    search = search_store_for_kv(memory, query, top_k=top_k)
    timings.retrieval_s += search.elapsed_s
    timings.vector_search_s += search.search_s
    return search


def _compose_memory(
    context: RequestContext,
    sample: ConversationSample,
    device_result: DeviceSearchResult,
    timings: RequestTimings,
) -> tuple[dict[str, Any], list[int]]:
    """Gather the retrieved rows from the chunk store and compose them inside the scaffold."""
    composer = context.corpus.composer(sample.sample_id)
    store = context.corpus.chunk_store(sample.sample_id)
    row_map = context.retrieval[sample.sample_id].row_map
    header, footer = _scaffold_chunks(composer)
    # Compose launches asynchronous CUDA work; synchronize on both sides of
    # the timer so it measures GPU time, not just kernel-enqueue time.
    synchronize_cuda()
    started = time.perf_counter()
    if context.config.kv_store_backend == "gpu":
        selected = [fetch_device_chunk(store, device_result.stable_ids, row_map)]
        staged_ids: list[str] = []
    else:
        # The lookup map stays on the GPU; only the selected rows cross to the
        # host side to stage their pinned payloads.
        selected = fetch_device_chunks(
            store, composer.chunks, device_result.stable_ids, row_map, reverse=True
        )
        staged_ids = [chunk.chunk_id for chunk in selected]
        if not staged_ids:
            raise RuntimeError("Retrieval returned no facts for a kv_injection request.")
    chunks = [header, *selected, footer]
    kv_by_layer = composer.encoder.compose(chunks)
    memory_token_ids = [token for chunk in chunks for token in chunk.token_ids]
    synchronize_cuda()
    timings.compose_s += time.perf_counter() - started
    release_chunks(store, staged_ids)
    # Packed slices keep their backing slabs alive; the selected views die with
    # this frame, well before the corpus store closes.
    return kv_by_layer, memory_token_ids


def _register(
    namespace: str,
    memory_user_id: str,
    kv_by_layer: dict[str, Any],
    memory_token_ids: list[int],
    timings: RequestTimings,
) -> None:
    # Registration hands the composed memory to the GPU-resident namespace
    # registry (a dict insert, no transfer); timed for parity with the other
    # setup metrics.
    started = time.perf_counter()
    register_user_memory(
        namespace,
        user_id=memory_user_id,
        kv_by_layer=kv_by_layer,
        num_tokens=len(memory_token_ids),
        token_ids=memory_token_ids,
    )
    synchronize_cuda()
    timings.store_write_s += time.perf_counter() - started


def _verify(
    context: RequestContext,
    memory: Any,
    sample: ConversationSample,
    device_result: DeviceSearchResult,
    memory_token_ids: list[int],
    timings: RequestTimings,
) -> None:
    """The composed tokens must equal the prompt-injection stream of the same retrieved facts."""
    started = time.perf_counter()
    hits = materialize_store_device_result(memory, device_result)
    canonical = build_memory_prompt_token_ids(
        context.corpus.tokenizer,
        sample,
        hits,
        context_window=0,
        memory_scaffold=context.scaffold,
    )
    require_identical_token_ids(memory_token_ids, canonical.token_ids)
    timings.verify_s += time.perf_counter() - started


def _scaffold_chunks(composer: Any) -> tuple[KVChunk, KVChunk]:
    header, footer = composer.header_chunk, composer.footer_chunk
    if header is None or footer is None:
        raise RuntimeError("Cannot compose a request before the scaffold chunks are encoded.")
    return header, footer
