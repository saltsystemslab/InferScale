"""Precompute a handful of context chunks, then answer a question with KV injection.

Needs a CUDA host and OPENAI_API_KEY in the environment.
Every chunk's KV is encoded once; at query time the retrieved
chunks are composed and injected into vLLM's KV cache instead of being
re-read as prompt text. Each target uses up to two preceding chunks as an
encoding prefix; only the target chunk's KV is retained.
"""

from inferscale.v1 import Chunk, InferScale, InferScaleConfig

CONTEXT = [
    ("c1", "Alice moved to Berlin in March 2021 to join a robotics startup."),
    ("c2", "Alice's favorite weekend activity is hiking in the Harz mountains."),
    ("c3", "Bob, Alice's brother, teaches high-school chemistry in Leeds."),
    ("c4", "Alice adopted a grey cat named Miso in 2023."),
]


def main() -> None:
    config = InferScaleConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        top_k=2,
        context_window=2,
        vector_backend="exact",
    )
    with InferScale(config) as engine:
        stats = engine.precompute(Chunk(id=chunk_id, text=text) for chunk_id, text in CONTEXT)
        print(f"precomputed {stats.chunk_count} chunks, {stats.token_count} tokens")
        engine.start()
        result = engine.query("What is the name of Alice's cat?")
        print(result.text)
        print(f"ttft={result.ttft_ms:.1f} ms retrieved={[hit.id for hit in result.hits]}")


if __name__ == "__main__":
    main()
