from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "locomo_jasper_bench"


def test_openai_batch_judging_module_and_identifiers_are_removed() -> None:
    assert not (PACKAGE_ROOT / "openai_batch_judging.py").exists()

    source = "\n".join(path.read_text() for path in PACKAGE_ROOT.rglob("*.py"))
    assert "OpenAIResponsesJudgeClient" not in source
    assert "OpenAIResponsesBatchJudgeClient" not in source
    assert "openai_judge_batch" not in source

