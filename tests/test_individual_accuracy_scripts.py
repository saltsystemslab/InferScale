from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_CASES = (
    ("gpu0_topk5.sh", "0", 5),
    ("gpu1_topk10.sh", "1", 10),
    ("gpu2_topk20.sh", "2", 20),
    ("gpu3_topk50.sh", "3", 50),
)


@pytest.mark.parametrize(("script_name", "gpu", "top_k"), SCRIPT_CASES)
def test_individual_accuracy_script_emits_five_qwen3_runs(
    tmp_path: Path,
    script_name: str,
    gpu: str,
    top_k: int,
) -> None:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update({"BENCHMARK_RESULTS_ROOT": str(tmp_path), "DRY_RUN": "1"})

    completed = subprocess.run(
        ["bash", str(ROOT / "scripts" / "individual" / script_name)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    # 4 KV windows plus the single prefix-qdrant prompt-injection baseline;
    # the prompt-injection-jasper baseline was removed from the sweep.
    assert "Sweep complete: 5 runs" in completed.stdout
    commands = [line for line in completed.stdout.splitlines() if "--run-id" in line]
    assert len(commands) == 5
    assert all(f"-k{top_k}-" in command for command in commands)
    assert {re.search(r"--answer-model ([^ ]+)", command).group(1) for command in commands} == {
        "qwen3-14b",
    }
    assert f'CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{gpu}}}"' in (
        ROOT / "scripts" / "individual" / script_name
    ).read_text()
