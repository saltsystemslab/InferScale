from __future__ import annotations

from locomo_jasper_bench.kv.answer_client import tokenize_messages
from locomo_jasper_bench.kv.chunked_rope import hit_turn_id, selected_turn_ids
from locomo_jasper_bench.vector_types import SearchHit


def _hit(rank: int, payload: dict) -> SearchHit:
    return SearchHit(id=f"h{rank}", payload=payload, score=1.0, distance=0.0, rank=rank)


def test_hit_turn_id_reads_metadata_first() -> None:
    hit = _hit(1, {"turn_id": "top-level", "metadata": {"turn_id": "metadata-id"}})

    assert hit_turn_id(hit) == "metadata-id"


def test_selected_turn_ids_preserves_rank_order_and_deduplicates() -> None:
    hits = [
        _hit(1, {"metadata": {"turn_id": "sample:session_1:0"}}),
        _hit(2, {"metadata": {"turn_id": "sample:session_1:1"}}),
        _hit(3, {"metadata": {"turn_id": "sample:session_1:0"}}),
        _hit(4, {"metadata": {}}),
    ]

    assert selected_turn_ids(hits) == ["sample:session_1:0", "sample:session_1:1"]


def test_tokenize_messages_uses_chat_template() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            return [len(messages), 7]

    assert tokenize_messages(FakeTokenizer(), [{"role": "user", "content": "hi"}]) == [1, 7]


def test_tokenize_messages_falls_back_to_encode() -> None:
    class FakeTokenizer:
        def encode(self, text):
            assert "USER: hi" in text
            return [3, 4, 5]

    assert tokenize_messages(FakeTokenizer(), [{"role": "user", "content": "hi"}]) == [3, 4, 5]

