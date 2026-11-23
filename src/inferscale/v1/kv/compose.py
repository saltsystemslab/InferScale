from __future__ import annotations

from collections.abc import Sequence

from ..types import KVChunk, ScaffoldChunks, SearchHit


def reverse_ranked_ids(hits: Sequence[SearchHit]) -> list[str]:
    best_first_ids = list(dict.fromkeys(str(hit.id) for hit in hits))
    return list(reversed(best_first_ids))


def memory_parts(scaffold: ScaffoldChunks, selected: Sequence[KVChunk]) -> list[KVChunk]:
    parts = [scaffold.header]
    parts.extend(selected if selected else [scaffold.empty])
    parts.append(scaffold.footer)
    return parts
