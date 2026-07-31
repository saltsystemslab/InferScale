"""QASPER dataset loader (Dasigi et al., 2021), official test split.

416 NLP papers with 1,451 questions. Each question targets one paper and has
several independent reference annotations; reference answer strings are
derived exactly like the official evaluator (qasper_evaluator.py, shipped in
the same archive): unanswerable -> "Unanswerable", extractive spans joined
with ", ", else the free-form answer, else Yes/No. String metrics take the
best over references.

The per-question type is the FIRST reference's type (extractive, abstractive,
boolean, or unanswerable; the official evaluator buckets by best-matching
reference instead and calls unanswerable "none"). Evidence is the union of
the references' evidence paragraphs, minus "FLOAT SELECTED" table and figure
entries, matching the evaluator's text_evidence_only mode; every evidence
paragraph appears verbatim in the rendered paper body, so fact_recall_at_k
measures whether retrieval surfaced an annotated evidence paragraph.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..data_types import (
    EvidenceRef,
    RagDocument,
    RagPromptProfile,
    RagQuery,
    doc_id_for_url,
)

TEST_FILENAME = "qasper-test-v0.3.json"
ARCHIVE_FILENAME = "qasper-test-and-evaluator-v0.3.tgz"
ARCHIVE_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
)

UNANSWERABLE_TEXT = "Unanswerable"
QASPER_PROMPT_PROFILE = RagPromptProfile(
    system_prompt=(
        "You are a helpful assistant that answers questions about scientific papers "
        "using retrieved excerpts from a corpus of NLP research papers. "
        "The following are excerpts from research papers:\n\n"
    ),
    answer_instruction=(
        "Answer the question using only the paper excerpts above.\n"
        "Answer with a short phrase, an exact span from the paper, or Yes or No. "
        f"If the excerpts do not contain the information needed, answer exactly: {UNANSWERABLE_TEXT}\n"
    ),
)

_ARXIV_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}(v\d+)?$")


def load_qasper(data_dir: Path) -> tuple[list[RagDocument], list[RagQuery]]:
    path = Path(data_dir) / TEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Download it with: RAG_DATASET=qasper "
            f"bash scripts/rag/setup_data.sh (source: {ARCHIVE_URL})."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a top-level JSON object keyed by paper id.")

    docs: list[RagDocument] = []
    queries: list[RagQuery] = []
    seen_question_ids: set[str] = set()
    for paper_id, paper in data.items():
        if not isinstance(paper, dict):
            raise ValueError(f"{path}: paper {paper_id} is not a JSON object.")
        title = str(paper.get("title") or "").strip()
        if not title:
            raise ValueError(f"{path}: paper {paper_id} has no title.")
        url = f"https://arxiv.org/abs/{paper_id}"
        doc = RagDocument(
            doc_id=doc_id_for_url(url),
            title=title,
            source="arXiv",
            published_at=_published_from_arxiv_id(str(paper_id)),
            category="NLP research paper",
            url=url,
            body=_render_paper_body(path, paper_id, paper),
            author=None,
        )
        docs.append(doc)
        for qa in paper.get("qas") or []:
            query = _parse_question(path, paper_id, doc, qa)
            if query.query_id in seen_question_ids:
                raise ValueError(f"{path}: duplicate question_id {query.query_id}.")
            seen_question_ids.add(query.query_id)
            queries.append(query)

    if not docs:
        raise ValueError(f"{path}: contains no papers.")
    if not queries:
        raise ValueError(f"{path}: contains no questions.")
    return docs, queries


def _parse_question(path: Path, paper_id: str, doc: RagDocument, qa: Any) -> RagQuery:
    if not isinstance(qa, dict):
        raise ValueError(f"{path}: paper {paper_id} has a non-object qa entry.")
    question = str(qa.get("question") or "").strip()
    question_id = str(qa.get("question_id") or "").strip()
    if not question or not question_id:
        raise ValueError(f"{path}: paper {paper_id} has a qa without question or question_id.")

    references: list[str] = []
    evidence_texts: dict[str, None] = {}
    first_type: str | None = None
    for annotation in qa.get("answers") or []:
        answer_info = (annotation or {}).get("answer")
        if not isinstance(answer_info, dict):
            raise ValueError(f"{path}: question {question_id} has a malformed annotation.")
        reference, reference_type, evidence = _derive_reference(path, question_id, answer_info)
        references.append(reference)
        if first_type is None:
            first_type = reference_type
        for text in evidence:
            evidence_texts.setdefault(text)
    if not references or first_type is None:
        raise ValueError(f"{path}: question {question_id} has no reference answers.")

    return RagQuery(
        query_id=question_id,
        question=question,
        gold_answers=tuple(references),
        question_type=first_type,
        evidence=tuple(
            EvidenceRef(doc_id=doc.doc_id, title=doc.title, url=doc.url, fact=fact)
            for fact in evidence_texts
        ),
    )


def _derive_reference(
    path: Path,
    question_id: str,
    answer_info: dict[str, Any],
) -> tuple[str, str, list[str]]:
    """Reference string, reference type, and text evidence for one annotation.

    Mirrors qasper_evaluator.get_answers_and_evidence with
    text_evidence_only=True, except the unanswerable type is named
    "unanswerable" instead of "none".
    """
    if answer_info.get("unanswerable"):
        return UNANSWERABLE_TEXT, "unanswerable", []
    if answer_info.get("extractive_spans"):
        reference = ", ".join(str(span) for span in answer_info["extractive_spans"])
        reference_type = "extractive"
    elif answer_info.get("free_form_answer"):
        reference = str(answer_info["free_form_answer"])
        reference_type = "abstractive"
    elif answer_info.get("yes_no"):
        reference = "Yes"
        reference_type = "boolean"
    elif answer_info.get("yes_no") is not None:
        reference = "No"
        reference_type = "boolean"
    else:
        raise ValueError(
            f"{path}: question {question_id} has an annotation without an answer."
        )
    evidence = [
        text.strip()
        for text in answer_info.get("evidence") or []
        if isinstance(text, str) and text.strip() and "FLOAT SELECTED" not in text
    ]
    return reference, reference_type, evidence


def _render_paper_body(path: Path, paper_id: str, paper: dict[str, Any]) -> str:
    parts: list[str] = []
    abstract = str(paper.get("abstract") or "").strip()
    if abstract:
        parts.append(abstract)
    for section in paper.get("full_text") or []:
        if not isinstance(section, dict):
            raise ValueError(f"{path}: paper {paper_id} has a malformed full_text section.")
        section_name = str(section.get("section_name") or "").strip()
        if section_name:
            parts.append(section_name)
        for paragraph in section.get("paragraphs") or []:
            text = str(paragraph or "").strip()
            if text:
                parts.append(text)
    body = "\n\n".join(parts)
    if not body:
        raise ValueError(f"{path}: paper {paper_id} rendered to an empty body.")
    return body


def _published_from_arxiv_id(paper_id: str) -> str:
    match = _ARXIV_ID_RE.match(paper_id.strip())
    if match is None:
        return "unknown"
    year = 2000 + int(match.group(1))
    return f"{year}-{match.group(2)}"


def _spec():
    from . import RagDatasetSpec

    return RagDatasetSpec(
        name="qasper",
        corpus_filename=TEST_FILENAME,
        queries_filename=TEST_FILENAME,
        download_urls={ARCHIVE_FILENAME: ARCHIVE_URL},
        chunking="token-window",
        prompt_profile=QASPER_PROMPT_PROFILE,
        load=load_qasper,
    )


QASPER_SPEC = _spec()
