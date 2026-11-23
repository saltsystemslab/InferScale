from __future__ import annotations

from benchmarks.rag.chunking import chunk_document, render_document
from benchmarks.rag.data_types import RagDocument

DOC = RagDocument(
    doc_id="d1",
    title="A title",
    source="A source",
    published_at="2024-01",
    category="news",
    url="https://example.com/a",
    body="The body of the document.",
)


class TokenizerWithOnlyEncodeAndDecode:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_chunking_needs_only_the_encode_and_decode_calls_vllm_and_transformers_tokenizers_share():
    chunks = chunk_document(DOC, tokenizer=TokenizerWithOnlyEncodeAndDecode(), chunk_size=16)

    assert "".join(chunk.text for chunk in chunks) == render_document(DOC)
    assert [chunk.chunk_id for chunk in chunks] == [f"d1:{index}" for index in range(len(chunks))]
    assert all(chunk.token_count == 16 for chunk in chunks[:-1])
