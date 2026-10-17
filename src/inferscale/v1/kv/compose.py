"""Ordering of retrieved chunks inside the composed memory."""

from __future__ import annotations

from collections.abc import Sequence

from ..types import KVChunk, ScaffoldChunks, SearchHit


def reverse_ranked_ids(hits: Sequence[SearchHit]) -> list[str]:
    """Unique retrieved ids with the best-ranked chunk last, closest to the query."""
    unique_ids = list(dict.fromkeys(str(hit.id) for hit in hits))
    return list(reversed(unique_ids))


def memory_parts(scaffold: ScaffoldChunks, selected: Sequence[KVChunk]) -> list[KVChunk]:
    """``[header, *selected, footer]``, or the empty-context chunk when nothing was retrieved."""
    parts = [scaffold.header]
    parts.extend(selected if selected else [scaffold.empty])
    parts.append(scaffold.footer)
    return parts
