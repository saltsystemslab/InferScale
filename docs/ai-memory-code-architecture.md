# ai-memory-code Architecture

This repository uses `ai-memory-code` in one opt-in answer path: strict GPU KV
injection for LoCoMo question answering. The normal benchmark path still uses
retrieved memory as prompt text; the KV path retrieves the same top-k turns, then
turns only those retrieved turns into GPU-resident KV tensors for vLLM.

## Component And Data Flow

```mermaid
flowchart LR
    cli["locomo-jasper-bench<br/>src/locomo_jasper_bench/run.py"]
    cfg["BenchmarkConfig<br/>answer_backend, vector_backend, KV options"]
    dataset["LoCoMo dataset<br/>data/locomo10.json"]
    runner["run_benchmark<br/>runner.py"]
    builder["SampleMemoryBuilder<br/>memory_builder.py"]
    mem0["Mem0 Memory<br/>infer=false turn ingestion"]
    cache["CachedEmbedder<br/>embedding cache"]
    adapter["Mem0JasperVectorStore<br/>mem0_adapter.py"]
    jasper["JasperVectorStore<br/>jasperpy GPU graph"]
    qdrant["QdrantVectorStore<br/>local qdrant-client"]
    search["QuestionEvaluator<br/>embed query + top-k search"]
    promptAnswer["OpenAICompatibleChatClient<br/>retrieved text in prompt"]
    kvAnswer["VLLMChunkedKVAnswerClient<br/>strict GPU KV path"]
    judge["Judge client<br/>OpenAI-compatible chat"]
    results["Run artifacts<br/>config.json, system.json,<br/>predictions.jsonl, summary.json"]

    cli --> cfg --> runner
    dataset --> runner
    runner --> builder
    builder --> mem0
    mem0 <--> cache
    mem0 --> adapter
    adapter -->|--vector-backend jasper| jasper
    adapter -->|--vector-backend qdrant| qdrant
    jasper --> search
    qdrant --> search
    search -->|SearchHit list with turn metadata| promptAnswer
    search -->|SearchHit list with turn metadata| kvAnswer
    promptAnswer --> judge
    kvAnswer --> judge
    judge --> results

    subgraph NormalAnswerPath["answer_backend=openai"]
        promptAnswer
    end

    subgraph StrictGpuKvPath["answer_backend=vllm-kv"]
        kvAnswer
    end
```

The vector retrieval step is shared by both answer modes. Each LoCoMo turn is
inserted into Mem0 with metadata such as `sample_id`, `session_id`, `turn_id`,
speaker, timestamp, and turn index. Retrieval returns `SearchHit` objects whose
metadata is later used by the KV path to select the exact conversation turns to
encode.

In normal mode, `QuestionEvaluator.answer_from_hits()` formats retrieved memory
text into chat messages through `build_retrieval_answer_messages()`. In strict
GPU KV mode, the same evaluator detects that the answer client exposes
`answer_with_retrieved_memory()` and delegates to the KV client instead of
building a prompt with memory text.

## Strict GPU KV Injection Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Runner as _run_kv_prediction_mode<br/>runner.py
    participant Builder as SampleMemoryBuilder
    participant Store as Mem0JasperVectorStore
    participant Search as QuestionEvaluator
    participant KVClient as VLLMChunkedKVAnswerClient
    participant Composer as ChunkedRopeSampleComposer
    participant PreRoPE as ai-memory-code/chunked-rope<br/>PreRoPEMemoryEncoder
    participant Rope as ai-memory-code/chunked-rope<br/>rope_inject.py
    participant Registry as strict_gpu_registry
    participant GStore as ai-memory-code<br/>GPUMemoryStore
    participant Connector as strict_gpu_connector.MemoryKVConnector
    participant Upstream as ai-memory-code<br/>MemoryKVConnector
    participant VLLM as in-process vLLM LLM

    Runner->>Builder: build(sample)
    Builder->>Store: insert LoCoMo turns + finalize index
    loop each question in sample
        Runner->>Search: _search_mem0_memory(memory, question)
        Search->>Store: search(query embedding, top_k)
        Store-->>Search: SearchHit list with turn_id metadata
        Search-->>Runner: hits + SearchMetrics
    end
    Runner->>Builder: close(memory)
    Note over Runner,Store: Vector index is closed before encoder/vLLM load.

    Runner->>KVClient: prepare_sample(sample, hits_by_question)
    KVClient->>Composer: encode_sample(sample, needed_turn_ids)
    Composer->>PreRoPE: load HF model and capture pre-RoPE K + V
    Composer->>Rope: extract model RoPE cos/sin table
    PreRoPE-->>Composer: EncodedChunk per retrieved turn
    KVClient->>VLLM: construct LLM(kv_transfer_config)
    VLLM->>Connector: import locomo_jasper_bench.kv.strict_gpu_connector
    Connector->>Upstream: subclass MemoryKVConnector
    Connector->>Registry: get_gpu_memory_store(namespace)
    Registry-->>Connector: process-local GPUMemoryStore

    loop each question
        Runner->>KVClient: answer_with_retrieved_memory(sample, qa, hits)
        KVClient->>Composer: compose(hits)
        Composer->>Rope: rotate chunks to contiguous virtual positions
        Rope-->>Composer: post-RoPE K tensors + V tensors
        Composer-->>KVClient: kv_by_layer, token_ids, num_tokens
        KVClient->>Registry: register_user_memory(namespace, user_id, KV)
        Registry->>GStore: add_user_memory(user_id, kv_by_layer, token_ids)
        KVClient->>VLLM: generate(prompt_token_ids = memory token_ids + query tokens)
        VLLM->>Upstream: get_num_new_matched_tokens(request)
        Upstream->>GStore: match prompt prefix against memory token_ids
        VLLM->>Upstream: update_state_after_alloc() + build_connector_meta()
        VLLM->>Upstream: start_load_kv(forward_context)
        Upstream->>GStore: get_user_memory(user_id)
        Upstream->>VLLM: scatter-copy KV into paged cache blocks
        VLLM-->>KVClient: generated answer text
        KVClient->>Registry: remove_user_memory(namespace, user_id)
    end

    Runner->>KVClient: close_sample()
    KVClient->>Registry: clear_namespace(namespace)
```

The local `strict_gpu_connector.MemoryKVConnector` intentionally keeps the
upstream connector mechanics but swaps the memory store to the process-local
registry namespace. It also forbids `memory_path` disk loads, so the strict path
does not use safetensors, `CPUMemoryStore`, vLLM CPU swap, or vLLM CPU offload.

## Boundaries And Responsibilities

- `benchmark-jasper` owns the benchmark loop, LoCoMo parsing, Mem0 ingestion,
  vector-store selection, answer/judge clients, result files, and the strict
  wrapper that wires `ai-memory-code` into an in-process vLLM run.
- `jasperpy` provides the GPU ANN graph used by `JasperVectorStore` when
  `--vector-backend jasper` is selected.
- `ai-memory-code/chunked-rope` provides the pre-RoPE memory encoder and RoPE
  rotation helpers used to compose retrieved chunks at request time.
- `ai-memory-code/memory_connector` provides the vLLM KV connector and
  `GPUMemoryStore` implementation that scatter-copy precomputed KV tensors into
  vLLM's paged attention cache.

## Runtime Modes

| Mode | Answer backend | Retrieval | Answer context |
| --- | --- | --- | --- |
| Baseline | `openai` | Mem0 + Jasper/Qdrant top-k | Retrieved memory text is inserted into the prompt. |
| Strict GPU KV | `vllm-kv` | Mem0 + Jasper/Qdrant top-k | Retrieved turns become GPU KV tensors injected before query tokens. |

Strict GPU KV mode currently supports `--kv-sample-window 1`. For each sample,
the benchmark retrieves all planned question memories, closes the vector index,
encodes the union of retrieved turn IDs, constructs in-process vLLM, answers the
sample's questions, and then clears the namespace before moving on.

