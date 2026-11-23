from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from benchmarks.rag.data_types import (
    UNANSWERABLE_TYPE,
    EvidenceRef,
    RagDatasetSpec,
    RagDocument,
    RagPromptProfile,
    RagQuery,
    doc_id_for_url,
)

TEST_FILENAME = "qasper-test-v0.3.json"
ARCHIVE_FILENAME = "qasper-test-and-evaluator-v0.3.tgz"
ARCHIVE_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

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
        "If the excerpts do not contain the information needed, answer exactly: "
        f"{UNANSWERABLE_TEXT}\n"
    ),
)

_ARXIV_ID_RE = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}(v\d+)?$")
ARXIV_ID_CENTURY = 2000
FLOAT_EVIDENCE_MARKER = "FLOAT SELECTED"


def load_qasper(data_dir: Path) -> tuple[list[RagDocument], list[RagQuery]]:
    path = Path(data_dir) / TEST_FILENAME
    papers = _load_papers(path)

    docs: list[RagDocument] = []
    queries: list[RagQuery] = []
    seen_question_ids: set[str] = set()
    for paper_id, paper in papers.items():
        doc = _document(path, paper_id, paper)
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


def _load_papers(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run bash scripts/rag/setup_data.sh to download it "
            f"(source: {ARCHIVE_URL})."
        )
    papers = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(papers, dict):
        raise ValueError(f"{path}: expected a top-level JSON object keyed by paper id.")
    return papers


def _document(path: Path, paper_id: str, paper: Any) -> RagDocument:
    if not isinstance(paper, dict):
        raise ValueError(f"{path}: paper {paper_id} is not a JSON object.")
    title = str(paper.get("title") or "").strip()
    if not title:
        raise ValueError(f"{path}: paper {paper_id} has no title.")
    url = f"https://arxiv.org/abs/{paper_id}"
    return RagDocument(
        doc_id=doc_id_for_url(url),
        title=title,
        source="arXiv",
        published_at=_published_from_arxiv_id(str(paper_id)),
        category="NLP research paper",
        url=url,
        body=_render_paper_body(path, paper_id, paper),
        author=None,
    )


class _Reference(NamedTuple):
    text: str
    question_type: str
    evidence: list[str]


def _parse_question(path: Path, paper_id: str, doc: RagDocument, qa: Any) -> RagQuery:
    if not isinstance(qa, dict):
        raise ValueError(f"{path}: paper {paper_id} has a non-object qa entry.")
    question = str(qa.get("question") or "").strip()
    question_id = str(qa.get("question_id") or "").strip()
    if not question or not question_id:
        raise ValueError(f"{path}: paper {paper_id} has a qa without question or question_id.")

    references = [
        _reference(path, question_id, annotation) for annotation in qa.get("answers") or []
    ]
    if not references:
        raise ValueError(f"{path}: question {question_id} has no reference answers.")
    first_reference = references[0]
    distinct_evidence = dict.fromkeys(
        paragraph for reference in references for paragraph in reference.evidence
    )
    return RagQuery(
        query_id=question_id,
        question=question,
        gold_answers=tuple(reference.text for reference in references),
        question_type=first_reference.question_type,
        evidence=tuple(
            EvidenceRef(doc_id=doc.doc_id, title=doc.title, url=doc.url, fact=fact)
            for fact in distinct_evidence
        ),
    )


def _reference(path: Path, question_id: str, annotation: Any) -> _Reference:
    answer_info = (annotation or {}).get("answer")
    if not isinstance(answer_info, dict):
        raise ValueError(f"{path}: question {question_id} has a malformed annotation.")
    if answer_info.get("unanswerable"):
        return _Reference(UNANSWERABLE_TEXT, UNANSWERABLE_TYPE, [])
    text, question_type = _answer_and_type(path, question_id, answer_info)
    return _Reference(text, question_type, _evidence_paragraphs(answer_info))


def _answer_and_type(path: Path, question_id: str, answer_info: dict[str, Any]) -> tuple[str, str]:
    if answer_info.get("extractive_spans"):
        return ", ".join(str(span) for span in answer_info["extractive_spans"]), "extractive"
    if answer_info.get("free_form_answer"):
        return str(answer_info["free_form_answer"]), "abstractive"
    if answer_info.get("yes_no"):
        return "Yes", "boolean"
    if answer_info.get("yes_no") is not None:
        return "No", "boolean"
    raise ValueError(f"{path}: question {question_id} has an annotation without an answer.")


def _evidence_paragraphs(answer_info: dict[str, Any]) -> list[str]:
    return [
        text.strip()
        for text in answer_info.get("evidence") or []
        if isinstance(text, str) and text.strip() and FLOAT_EVIDENCE_MARKER not in text
    ]


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
    two_digit_year, month = match.group(1), match.group(2)
    return f"{ARXIV_ID_CENTURY + int(two_digit_year)}-{month}"


QASPER_SPEC = RagDatasetSpec(
    name="qasper",
    corpus_filename=TEST_FILENAME,
    queries_filename=TEST_FILENAME,
    download_urls={ARCHIVE_FILENAME: ARCHIVE_URL},
    prompt_profile=QASPER_PROMPT_PROFILE,
    load=load_qasper,
)
