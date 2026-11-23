from __future__ import annotations

from rag_fixtures import qasper_test_records, write_qasper_files

from benchmarks.rag.datasets.qasper import load_qasper

FIRST_PAPER = "1601.00001"


def load(tmp_path, records):
    return load_qasper(write_qasper_files(tmp_path / "data", records))


def paper_with(**changes) -> dict:
    records = qasper_test_records()
    records[FIRST_PAPER].update(changes)
    return records


def test_paper_body_is_the_abstract_then_each_section_title_and_paragraph(tmp_path):
    records = paper_with(
        abstract="  The abstract.  ",
        full_text=[
            {"section_name": " Introduction ", "paragraphs": [" First. ", "", None, "Second."]},
            {"section_name": "", "paragraphs": ["Untitled section text."]},
            {"section_name": "Empty section", "paragraphs": None},
        ],
    )

    docs, _ = load(tmp_path, records)

    assert docs[0].body == (
        "The abstract.\n\nIntroduction\n\nFirst.\n\nSecond.\n\n"
        "Untitled section text.\n\nEmpty section"
    )
    assert (docs[0].source, docs[0].category, docs[0].author) == (
        "arXiv",
        "NLP research paper",
        None,
    )
