"""Shared stubs and fixtures for the rag_bench test files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CharTokenizer:
    """Lossless char-level tokenizer: encode is ord per char, decode is chr join."""

    bos_token_id = None

    def encode(self, text: str, **_: Any) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class TemplateTokenizer(CharTokenizer):
    """Char tokenizer with a content-trimming chat template (Llama-style)."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> Any:
        text = "".join(
            f"<{message['role']}>{message['content'].strip()}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


FACT_A = "Alpha Corp announced a breakthrough in perovskite solar panels"
FACT_A2 = "Alpha Corp expects mass production to begin next spring"
FACT_B = "quarterly profits doubled at Beta Industries after the merger"
FACT_C = "Gamma LLC delayed its flagship product launch until December"

_DOC_A = {
    "title": "Alpha Corp stuns the energy market",
    "author": "Jo Reporter",
    "body": (
        f"In a surprise event, {FACT_A}. Analysts called the demonstration convincing. "
        f"During the same briefing, {FACT_A2}, according to its chief executive. "
        "Rivals declined to comment on the announcement."
    ),
    "category": "technology",
    "published_at": "2023-10-01T08:00:00+00:00",
    "source": "The Example Times",
    "url": "https://example.com/alpha-corp-solar",
}
_DOC_B = {
    "title": "Beta Industries posts record quarter",
    "author": None,
    "body": (
        f"Regulators approved the deal in September, and {FACT_B}. "
        "The company raised its guidance for the rest of the fiscal year. "
        "Employees will receive a one-time bonus, the board said."
    ),
    "category": "business",
    "published_at": "2023-10-15T12:30:00+00:00",
    "source": "Example Business Daily",
    "url": "https://example.com/beta-industries-quarter",
}
_DOC_C = {
    "title": "Gamma LLC slips its launch window",
    "author": "Sam Writer",
    "body": (
        f"Citing supply constraints, {FACT_C}. "
        "Preorder customers will be offered refunds or store credit. "
        "The company blamed a shortage of custom display panels."
    ),
    "category": "technology",
    "published_at": "2023-11-02T09:15:00+00:00",
    "source": "The Example Times",
    "url": "https://example.com/gamma-llc-delay",
}


def multihop_corpus_records() -> list[dict[str, Any]]:
    return [dict(_DOC_A), dict(_DOC_B), dict(_DOC_C)]


def multihop_query_records() -> list[dict[str, Any]]:
    return [
        {
            "query": "Which company announced a solar breakthrough while Beta Industries doubled profits?",
            "answer": "Alpha Corp",
            "question_type": "inference_query",
            "evidence_list": [
                _evidence(_DOC_A, FACT_A),
                _evidence(_DOC_B, FACT_B),
            ],
        },
        {
            "query": "Did both Alpha Corp and Gamma LLC make announcements about future timelines?",
            "answer": "Yes",
            "question_type": "comparison_query",
            "evidence_list": [
                _evidence(_DOC_A, FACT_A2),
                _evidence(_DOC_C, FACT_C),
            ],
        },
        {
            "query": "After the merger approval in September, what happened at Beta Industries?",
            "answer": "Profits doubled",
            "question_type": "temporal_query",
            # No url on purpose: exercises the exact-title fallback resolution.
            "evidence_list": [
                {
                    "title": _DOC_B["title"],
                    "author": None,
                    "category": _DOC_B["category"],
                    "fact": FACT_B,
                    "published_at": _DOC_B["published_at"],
                    "source": _DOC_B["source"],
                    "url": "",
                }
            ],
        },
        {
            "query": "What color is the Alpha Corp headquarters lobby?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]


def write_multihop_files(
    data_dir: Path,
    corpus: list[dict[str, Any]] | None = None,
    queries: list[dict[str, Any]] | None = None,
) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "corpus.json").write_text(
        json.dumps(multihop_corpus_records() if corpus is None else corpus),
        encoding="utf-8",
    )
    (data_dir / "MultiHopRAG.json").write_text(
        json.dumps(multihop_query_records() if queries is None else queries),
        encoding="utf-8",
    )
    return data_dir


def _evidence(doc: dict[str, Any], fact: str) -> dict[str, Any]:
    return {
        "title": doc["title"],
        "author": doc["author"],
        "category": doc["category"],
        "fact": fact,
        "published_at": doc["published_at"],
        "source": doc["source"],
        "url": doc["url"],
    }


def _qasper_answer(**overrides: Any) -> dict[str, Any]:
    answer = {
        "unanswerable": False,
        "extractive_spans": [],
        "yes_no": None,
        "free_form_answer": "",
        "evidence": [],
        "highlighted_evidence": [],
    }
    answer.update(overrides)
    return {"annotation_id": "a", "worker_id": "w", "answer": answer}


QASPER_EVIDENCE_RESULTS = "Our best model reaches 42.0 F1 on the probing benchmark."
QASPER_EVIDENCE_TOOLKIT = "We release our probing toolkit publicly."
QASPER_EVIDENCE_SURVEY = "The survey covers twelve datasets across four languages."


def qasper_test_records() -> dict[str, Any]:
    return {
        "1601.00001": {
            "title": "Paper One: Attention Study",
            "abstract": "We study attention mechanisms in encoders.",
            "full_text": [
                {
                    "section_name": "Introduction",
                    "paragraphs": [
                        "Attention weights concentrate on rare tokens in our probing study.",
                        QASPER_EVIDENCE_TOOLKIT,
                    ],
                },
                {
                    "section_name": "Results",
                    "paragraphs": [QASPER_EVIDENCE_RESULTS, ""],
                },
            ],
            "qas": [
                {
                    "question": "What F1 does the best model reach?",
                    "question_id": "q-extractive",
                    "answers": [
                        _qasper_answer(
                            extractive_spans=["42.0 F1", "on the probing benchmark"],
                            evidence=[
                                QASPER_EVIDENCE_RESULTS,
                                "FLOAT SELECTED: Table 2: Results",
                            ],
                        ),
                        _qasper_answer(
                            free_form_answer="42 F1",
                            evidence=[QASPER_EVIDENCE_RESULTS],
                        ),
                    ],
                },
                {
                    "question": "Is the probing toolkit public?",
                    "question_id": "q-boolean",
                    "answers": [
                        _qasper_answer(yes_no=True, evidence=[QASPER_EVIDENCE_TOOLKIT]),
                        _qasper_answer(yes_no=False),
                    ],
                },
            ],
        },
        "1707.99999": {
            "title": "Paper Two: Datasets Survey",
            "abstract": "A survey of QA datasets.",
            "full_text": [
                {"section_name": "", "paragraphs": [QASPER_EVIDENCE_SURVEY]},
            ],
            "qas": [
                {
                    "question": "How many annotators were hired?",
                    "question_id": "q-unanswerable",
                    "answers": [
                        _qasper_answer(unanswerable=True),
                        _qasper_answer(
                            free_form_answer="Twelve annotators",
                            evidence=[QASPER_EVIDENCE_SURVEY],
                        ),
                    ],
                },
            ],
        },
    }


def write_qasper_files(data_dir: Path, records: dict[str, Any] | None = None) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "qasper-test-v0.3.json").write_text(
        json.dumps(qasper_test_records() if records is None else records),
        encoding="utf-8",
    )
    return data_dir
