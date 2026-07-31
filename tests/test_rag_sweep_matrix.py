from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Sweep axes the tests control explicitly; anything inherited from the
# developer's shell would otherwise change the default grids under test.
_CONTROLLED_ENV_VARS = (
    "RAG_DATASETS",
    "RAG_DATASET",
    "MODELS",
    "RAG_MODELS_QASPER",
    "RAG_MODELS_MULTIHOPRAG",
    "TOPKS",
    "RAG_WINDOW",
    "RAG_CHUNK_SIZE",
    "RUN_STAMP",
)

_JUDGE_ENV = {
    "JUDGE_BASE_URL": "http://judge.invalid",
    "JUDGE_API_KEY": "secret",
    "JUDGE_MODEL": "judge-model",
}


def _run_rag_script(script: str, results_root: Path, **overrides: str) -> str:
    env = os.environ.copy()
    for name in _CONTROLLED_ENV_VARS:
        env.pop(name, None)
    env.update({"BENCHMARK_RESULTS_ROOT": str(results_root), "DRY_RUN": "1"})
    env.update(overrides)
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "rag" / script)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout


def _expected_default_run_ids(stamp: str) -> set[str]:
    return {
        f"llama-kv-multihoprag-c1024-w5-k15-{stamp}",
        f"llama-prefix-multihoprag-c1024-w5-k15-{stamp}",
        f"qwen-kv-qasper-c1024-w5-k15-{stamp}",
        f"qwen-prefix-qasper-c1024-w5-k15-{stamp}",
    }


def _judge_run_ids(output: str) -> set[str]:
    run_id_block = output.split("Run-ids:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    return {line.strip() for line in run_id_block.splitlines() if line.strip()}


def test_full_run_defaults_emit_per_dataset_models(tmp_path: Path) -> None:
    stamp = "20260731T120000Z"
    output = _run_rag_script("full_run.sh", tmp_path, RUN_STAMP=stamp)

    assert f"RAG sweep complete: 4 runs (stamp {stamp})" in output
    run_ids = re.findall(r"--run-id ([^\s]+)", output)
    assert len(run_ids) == 4
    assert set(run_ids) == _expected_default_run_ids(stamp)

    commands = [line for line in output.splitlines() if "--run-id" in line]
    for command in commands:
        if "--dataset-name multihoprag" in command:
            assert "--answer-model llama" in command
        if "--dataset-name qasper" in command:
            assert "--answer-model qwen" in command
    assert sum("--answer-backend vllm-kv" in command for command in commands) == 2
    assert sum("--answer-backend vllm-prefix" in command for command in commands) == 2
    assert all("--skip-judge" in command for command in commands)


def test_full_run_models_override_applies_to_every_dataset(tmp_path: Path) -> None:
    stamp = "20260731T120000Z"
    output = _run_rag_script("full_run.sh", tmp_path, RUN_STAMP=stamp, MODELS="qwen")

    assert f"RAG sweep complete: 4 runs (stamp {stamp})" in output
    assert set(re.findall(r"--run-id ([^\s]+)", output)) == {
        f"qwen-kv-multihoprag-c1024-w5-k15-{stamp}",
        f"qwen-prefix-multihoprag-c1024-w5-k15-{stamp}",
        f"qwen-kv-qasper-c1024-w5-k15-{stamp}",
        f"qwen-prefix-qasper-c1024-w5-k15-{stamp}",
    }


def test_full_run_honors_legacy_single_dataset_env(tmp_path: Path) -> None:
    stamp = "20260731T120000Z"
    output = _run_rag_script("full_run.sh", tmp_path, RUN_STAMP=stamp, RAG_DATASET="qasper")

    assert f"RAG sweep complete: 2 runs (stamp {stamp})" in output
    assert set(re.findall(r"--run-id ([^\s]+)", output)) == {
        f"qwen-kv-qasper-c1024-w5-k15-{stamp}",
        f"qwen-prefix-qasper-c1024-w5-k15-{stamp}",
    }


def test_full_run_per_dataset_model_env_expands_only_that_dataset(tmp_path: Path) -> None:
    stamp = "20260731T120000Z"
    output = _run_rag_script(
        "full_run.sh",
        tmp_path,
        RUN_STAMP=stamp,
        RAG_MODELS_QASPER="llama qwen",
    )

    assert f"RAG sweep complete: 6 runs (stamp {stamp})" in output
    run_ids = set(re.findall(r"--run-id ([^\s]+)", output))
    assert run_ids == _expected_default_run_ids(stamp) | {
        f"llama-kv-qasper-c1024-w5-k15-{stamp}",
        f"llama-prefix-qasper-c1024-w5-k15-{stamp}",
    }


def test_judge_grid_matches_full_run_defaults(tmp_path: Path) -> None:
    stamp = "20260731T120000Z"
    output = _run_rag_script(
        "judge.sh",
        tmp_path,
        RUNIDS_FROM="grid",
        STAMP=stamp,
        **_JUDGE_ENV,
    )

    assert f"Would judge 4 RAG run(s) for stamp {stamp} (source: grid)." in output
    assert _judge_run_ids(output) == _expected_default_run_ids(stamp)


def test_judge_discovery_finds_both_datasets_and_ignores_foreign_runs(tmp_path: Path) -> None:
    stamp = "20260731T120000Z"
    expected = _expected_default_run_ids(stamp)
    decoys = (
        # LoCoMo run-id shape must never be judged by the RAG judge script.
        f"llama-kv-mem0-jasper10-k5-s0-{stamp}",
        # Unknown dataset names are excluded by the alternation.
        f"llama-kv-otherds-c1024-w5-k15-{stamp}",
    )
    for run_id in (*expected, *decoys):
        (tmp_path / run_id).mkdir()

    output = _run_rag_script(
        "judge.sh",
        tmp_path,
        RUNIDS_FROM="discover",
        STAMP=stamp,
        **_JUDGE_ENV,
    )

    assert f"Would judge 4 RAG run(s) for stamp {stamp} (source: discover)." in output
    assert _judge_run_ids(output) == expected
    assert all(decoy not in _judge_run_ids(output) for decoy in decoys)
