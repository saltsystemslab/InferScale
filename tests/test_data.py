from __future__ import annotations

import json

from locomo_jasper_bench.data import format_turn_for_memory, load_locomo


def test_load_locomo_sessions_and_qa(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "conv-1",
                    "conversation": {
                        "session_2_date_time": "2024-01-02",
                        "session_2": [{"speaker": "Bob", "text": "I moved to Boston."}],
                        "session_1_date_time": "2024-01-01",
                        "session_1": [
                            {"speaker": "Alice", "text": "I adopted Pixel.", "blip_caption": "a black cat"}
                        ],
                    },
                    "qa": [
                        {
                            "question_id": "q1",
                            "question": "Who adopted Pixel?",
                            "answer": "Alice",
                            "category": "single-hop",
                            "evidence": ["session_1"],
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    samples = load_locomo(path)

    assert len(samples) == 1
    assert samples[0].sample_id == "conv-1"
    assert [turn.session_id for turn in samples[0].turns] == ["session_1", "session_2"]
    assert samples[0].qa[0].question == "Who adopted Pixel?"
    assert samples[0].qa[0].category == "single-hop"
    assert "Image caption: a black cat" in format_turn_for_memory(samples[0].turns[0])
