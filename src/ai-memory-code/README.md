# MemoryKVConnector — GPU-Native AI Memory for vLLM

A vLLM KV Connector plugin that injects pre-computed user memory KV caches
directly into the paged attention system, bypassing prompt injection entirely.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Memory Encoding (offline)                    │
│                                                                  │
│  Memory Text ──→ Tokenize ──→ Model Forward ──→ Extract KV      │
│  "User likes..."   tokens      (HuggingFace)     DynamicCache   │
│                                                       │          │
│                                        Reshape to vLLM format    │
│                                     [2, num_tokens, heads, dim]  │
│                                               │                  │
│                                    ┌──────────▼──────────┐       │
│                                    │   GPUMemoryStore     │       │
│                                    │  (GPU-resident, per  │       │
│                                    │   user KV tensors)   │       │
│                                    └──────────┬──────────┘       │
└───────────────────────────────────────────────┼──────────────────┘
                                                │
┌───────────────────────────────────────────────┼──────────────────┐
│                  vLLM Serving (online)         │                  │
│                                                │                  │
│  ┌─────────────────────────────┐               │                  │
│  │ Scheduler Side              │               │                  │
│  │                             │               │                  │
│  │ get_num_new_matched_tokens()│               │                  │
│  │  → "user_123 has 512 tokens │               │                  │
│  │     of pre-computed KV"     │               │                  │
│  │                             │               │                  │
│  │ vLLM allocates 32 paged    │               │                  │
│  │ blocks for memory tokens    │               │                  │
│  │                             │               │                  │
│  │ build_connector_meta()      │               │                  │
│  │  → package block_ids +      │               │                  │
│  │    user_id for worker       │               │                  │
│  └──────────────┬──────────────┘               │                  │
│                 │ MemoryConnectorMetadata       │                  │
│                 ▼                               │                  │
│  ┌──────────────────────────────┐              │                  │
│  │ Worker Side                  │              │                  │
│  │                              │              │                  │
│  │ start_load_kv()              │◄─────────────┘                  │
│  │  → scatter-copy memory KV    │  (reads from GPUMemoryStore)   │
│  │    into allocated paged      │                                 │
│  │    blocks via slot_mapping   │                                 │
│  │                              │                                 │
│  │ Forward pass runs normally   │                                 │
│  │  → attention sees memory KV  │                                 │
│  │    in the paged cache        │                                 │
│  └──────────────────────────────┘                                 │
└───────────────────────────────────────────────────────────────────┘
```

## Complexity Advantage

| Approach | Prefill Complexity | Per-Request Overhead |
|----------|-------------------|---------------------|
| Prompt injection | O((m+q)²) | Re-encode m memory tokens every request |
| **KV injection** | **O(q·(m+q))** | **Scatter-copy m KV vectors (GPU memcpy)** |
| KV injection + ANN | **O(q·k)** | **Retrieve top-k + scatter-copy** |

Where m = memory tokens, q = query tokens, k = top-k retrieved memories.