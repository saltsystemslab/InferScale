from __future__ import annotations

import importlib
import json

import pytest

from benchmarks.rag.run import main


@pytest.mark.parametrize(
    "stage,module,function,summary",
    [
        ("estimate", "runner", "run_estimate", {}),
        (
            "preembed", "preembed", "preembed_rag_embeddings",
            {"cache": {"cache_dir": "cache", "hits": 1, "misses": 0},
             "chunk_embedding_count": 1, "query_embedding_count": 1},
        ),
        (
            "precompute-kv", "kv_precompute", "precompute_rag_kv",
            {"cache_dir": "cache", "encoded": 1, "skipped": 0, "chunk_count": 1, "cache_bytes": 4},
        ),
        ("run", "runner", "run_answer", {"question_count": 1, "judged_count": 0}),
        ("judge", "runner", "judge_existing_run", {"question_count": 1, "judged_count": 1}),
    ],
)
def test_main_dispatches_json_config_to_selected_stage(
    tmp_path, monkeypatch, stage, module, function, summary
) -> None:
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps({"models": {"answer": "configured/model"}}))
    run_path = tmp_path / "rag.json"
    run_path.write_text(json.dumps({"model": "answer", "run_id": "existing", "top_k": 3}))
    calls = []

    def run(config):
        calls.append(config)
        return summary

    monkeypatch.setattr(importlib.import_module(f"benchmarks.rag.{module}"), function, run)
    # Explicit runtime selection must work even if the ambient default is stale.
    monkeypatch.setenv("INFERSCALE_RUNTIME_CONFIG", "/missing/runtime.json")
    main(run_path, runtime_path=runtime_path, stage=stage)
    assert len(calls) == 1
    assert calls[0].model == "configured/model"
    assert calls[0].top_k == 3
