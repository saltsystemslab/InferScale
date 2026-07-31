from __future__ import annotations

import math

import pytest

from rag_bench.chunking import (
    chunk_corpus,
    chunk_document,
    chunks_by_doc,
    corpus_fingerprint,
    render_document,
)
from rag_bench.data_types import RagDocument, doc_id_for_url
from rag_test_utils import CharTokenizer


def _doc(url: str = "https://example.com/a", body: str | None = None) -> RagDocument:
    return RagDocument(
        doc_id=doc_id_for_url(url),
        title="A title",
        source="A source",
        published_at="2023-10-01T00:00:00+00:00",
        category="technology",
        url=url,
        body=body if body is not None else ("word " * 60).strip(),
    )


def test_render_prepends_metadata_header_once() -> None:
    rendered = render_document(_doc())

    assert rendered.startswith("Title: A title\nSource: A source\nPublished: ")
    assert rendered.count("Title: A title") == 1
    assert rendered.endswith("\n")


def test_chunk_boundaries_and_remainder() -> None:
    tokenizer = CharTokenizer()
    doc = _doc()
    chunk_size = 64
    rendered_ids = tokenizer.encode(render_document(doc))

    chunks = chunk_document(doc, tokenizer=tokenizer, chunk_size=chunk_size)

    assert len(chunks) == math.ceil(len(rendered_ids) / chunk_size)
    assert all(chunk.token_count == chunk_size for chunk in chunks[:-1])
    assert 1 <= chunks[-1].token_count <= chunk_size
    concatenated: list[int] = []
    for chunk in chunks:
        concatenated.extend(chunk.token_ids)
    assert concatenated == rendered_ids
    assert [chunk.chunk_id for chunk in chunks] == [
        f"{doc.doc_id}:{index}" for index in range(len(chunks))
    ]
    assert chunks[0].text.startswith("Title: A title")
    assert "Title:" not in chunks[1].text


def test_chunk_text_is_exact_decode_of_token_ids() -> None:
    tokenizer = CharTokenizer()
    chunks = chunk_document(_doc(), tokenizer=tokenizer, chunk_size=50)

    for chunk in chunks:
        assert chunk.text == tokenizer.decode(chunk.token_ids)
    assert "".join(chunk.text for chunk in chunks) == render_document(_doc())


def test_chunking_is_deterministic() -> None:
    tokenizer = CharTokenizer()
    first = chunk_document(_doc(), tokenizer=tokenizer, chunk_size=32)
    second = chunk_document(_doc(), tokenizer=tokenizer, chunk_size=32)

    assert [chunk.token_ids for chunk in first] == [chunk.token_ids for chunk in second]


def test_chunk_corpus_and_grouping() -> None:
    tokenizer = CharTokenizer()
    docs = [_doc("https://example.com/a"), _doc("https://example.com/b")]

    chunks = chunk_corpus(docs, tokenizer=tokenizer, chunk_size=48)
    grouped = chunks_by_doc(chunks)

    assert set(grouped) == {doc.doc_id for doc in docs}
    for doc_chunks in grouped.values():
        assert [chunk.chunk_index for chunk in doc_chunks] == list(range(len(doc_chunks)))


def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        chunk_document(_doc(), tokenizer=CharTokenizer(), chunk_size=0)


def test_corpus_fingerprint_is_content_and_order_sensitive() -> None:
    doc_a = _doc("https://example.com/a")
    doc_b = _doc("https://example.com/b")
    baseline = corpus_fingerprint([doc_a, doc_b])

    assert corpus_fingerprint([doc_a, doc_b]) == baseline
    assert corpus_fingerprint([doc_b, doc_a]) != baseline
    edited = _doc("https://example.com/a", body=doc_a.body + " extra")
    assert corpus_fingerprint([edited, doc_b]) != baseline
    assert len(baseline) == 20
