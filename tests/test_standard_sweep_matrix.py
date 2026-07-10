from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama", "mistral", "qwen", "qwen3-14b")
TOP_KS = (5, 10, 20, 50, 100)
WINDOWS = (0, 5, 20, 50)


def _expected_run_ids(stamp: str) -> set[str]:
    return {
        run_id
        for model in MODELS
        for top_k in TOP_KS
        for run_id in (
            *(
                f"{model}-kv-gpu-jasper10-k{top_k}-w{window}-{stamp}"
                for window in WINDOWS
            ),
            f"{model}-prefix-qdrant10-k{top_k}-{stamp}",
        )
    }


def _run_script(script: str, results_root: Path, **overrides: str) -> str:
    env = os.environ.copy()
    env.update(
        {
            "BENCHMARK_RESULTS_ROOT": str(results_root),
            "DRY_RUN": "1",
            "MODELS": " ".join(MODELS),
            "TOPKS": " ".join(str(top_k) for top_k in TOP_KS),
            "WINDOWS": " ".join(str(window) for window in WINDOWS),
        }
    )
    env.update(overrides)
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / script)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout


def _judge_run_ids(output: str) -> set[str]:
    run_id_block = output.split("Run-ids:\n", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
    return {line.strip() for line in run_id_block.splitlines() if line.strip()}


def test_full_run_dry_run_emits_the_100_run_standard_matrix(tmp_path: Path) -> None:
    output = _run_script("full_run.sh", tmp_path)
    stamp_match = re.search(r"Sweep complete: 100 runs \(stamp ([^)]+)\)", output)

    assert stamp_match is not None
    run_ids = re.findall(r"--run-id ([^\s]+)", output)
    assert len(run_ids) == 100
    assert set(run_ids) == _expected_run_ids(stamp_match.group(1))
    assert "prefix-gpu-jasper" not in output


def test_judge_grid_dry_run_emits_the_same_100_run_matrix(tmp_path: Path) -> None:
    stamp = "20260710T120000Z"
    output = _run_script(
        "judge.sh",
        tmp_path,
        RUNIDS_FROM="grid",
        STAMP=stamp,
        JUDGE_BASE_URL="http://judge.invalid",
        JUDGE_API_KEY="secret",
        JUDGE_MODEL="judge-model",
    )

    assert f"Would judge 100 run(s) for stamp {stamp} (source: grid)." in output
    assert _judge_run_ids(output) == _expected_run_ids(stamp)
    assert "prefix-gpu-jasper" not in output


def test_judge_discovery_ignores_legacy_jasper_prefix_runs(tmp_path: Path) -> None:
    stamp = "20260710T120000Z"
    expected = {
        f"llama-kv-gpu-jasper10-k5-w0-{stamp}",
        f"llama-prefix-qdrant10-k5-{stamp}",
    }
    legacy = f"llama-prefix-gpu-jasper10-k5-{stamp}"
    for run_id in (*expected, legacy):
        (tmp_path / run_id).mkdir()

    output = _run_script(
        "judge.sh",
        tmp_path,
        RUNIDS_FROM="discover",
        STAMP=stamp,
        JUDGE_BASE_URL="http://judge.invalid",
        JUDGE_API_KEY="secret",
        JUDGE_MODEL="judge-model",
    )

    assert f"Would judge 2 run(s) for stamp {stamp} (source: discover)." in output
    assert _judge_run_ids(output) == expected
    assert legacy not in output
