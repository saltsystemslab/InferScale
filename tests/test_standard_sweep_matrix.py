from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama", "mistral", "qwen", "qwen3-14b")
TOP_KS = (5, 10, 20, 50)
KV_WINDOWS = (0, 5, 20, 50)
INDIVIDUAL_RUN_SCRIPTS = {
    5: "individual/gpu0_topk5.sh",
    10: "individual/gpu1_topk10.sh",
    20: "individual/gpu2_topk20.sh",
    50: "individual/gpu3_topk50.sh",
}
INDIVIDUAL_JUDGE_SCRIPTS = {
    5: "individual/judge_gpu0_topk5.sh",
    10: "individual/judge_gpu1_topk10.sh",
    20: "individual/judge_gpu2_topk20.sh",
    50: "individual/judge_gpu3_topk50.sh",
}


def _expected_full_run_ids(stamp: str) -> set[str]:
    kv_ids = {
        f"{model}-kv-mem0-jasper10-k{top_k}-s{window}-{stamp}"
        for model in MODELS
        for top_k in TOP_KS
        for window in KV_WINDOWS
    }
    # Prompt injection runs only with the qdrant backend; the jasper
    # prompt-injection baseline was removed from the sweep.
    prefix_ids = {
        f"{model}-prefix-mem0-qdrant10-k{top_k}-s0-{stamp}"
        for model in MODELS
        for top_k in TOP_KS
    }
    return kv_ids | prefix_ids


def _run_script(script: str, results_root: Path, **overrides: str) -> str:
    env = os.environ.copy()
    env.update(
        {
            "BENCHMARK_RESULTS_ROOT": str(results_root),
            "DRY_RUN": "1",
            "MODELS": " ".join(MODELS),
            "TOPKS": " ".join(str(top_k) for top_k in TOP_KS),
            "KV_WINDOWS": " ".join(str(window) for window in KV_WINDOWS),
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


def test_full_run_dry_run_emits_the_80_run_mem0_fact_matrix(tmp_path: Path) -> None:
    output = _run_script("full_run.sh", tmp_path)
    stamp_match = re.search(r"Sweep complete: 80 runs \(stamp ([^)]+)\)", output)

    assert stamp_match is not None
    run_ids = re.findall(r"--run-id ([^\s]+)", output)
    assert len(run_ids) == 80
    assert set(run_ids) == _expected_full_run_ids(stamp_match.group(1))

    kv_commands = [line for line in output.splitlines() if "--answer-backend vllm-kv" in line]
    prefix_commands = [line for line in output.splitlines() if "--answer-backend vllm-prefix" in line]
    assert len(kv_commands) == 64
    assert len(prefix_commands) == 16
    for window in KV_WINDOWS:
        assert any(
            f"--context-window {window}" in command and f"-s{window}-" in command
            for command in kv_commands
        )
    assert not any(
        re.search(rf"--context-window {window}(?:\s|$)", command)
        for window in (1, 2, 3)
        for command in (*kv_commands, *prefix_commands)
    )
    assert all("-mem0-" in run_id for run_id in run_ids)
    assert all("--vector-backend jasper" in command for command in kv_commands)
    assert all("--vector-backend qdrant" in command for command in prefix_commands)
    assert all("--context-window 0" in command and "-s0-" in command for command in prefix_commands)


def test_full_run_honors_shared_run_stamp(tmp_path: Path) -> None:
    stamp = "20260711T220000Z"
    output = _run_script(
        "full_run.sh",
        tmp_path,
        RUN_STAMP=stamp,
        MODELS="qwen",
        TOPKS="5",
        KV_WINDOWS="0",
    )

    assert f"Sweep complete: 2 runs (stamp {stamp})" in output
    assert f"qwen-kv-mem0-jasper10-k5-s0-{stamp}" in output
    assert f"qwen-prefix-mem0-qdrant10-k5-s0-{stamp}" in output
    assert "prefix-mem0-jasper" not in output


def test_individual_judge_scripts_partition_the_shared_sweep(tmp_path: Path) -> None:
    stamp = "20260711T220000Z"
    combined_run_ids: set[str] = set()

    for top_k, script in INDIVIDUAL_JUDGE_SCRIPTS.items():
        output = _run_script(
            script,
            tmp_path,
            RUN_STAMP=stamp,
            JUDGE_BASE_URL="http://judge.invalid",
            JUDGE_API_KEY="secret",
            JUDGE_MODEL="judge-model",
        )
        run_ids = _judge_run_ids(output)
        expected = {
            *(
                f"qwen3-14b-kv-mem0-jasper10-k{top_k}-s{window}-{stamp}"
                for window in KV_WINDOWS
            ),
            f"qwen3-14b-prefix-mem0-qdrant10-k{top_k}-s0-{stamp}",
        }

        assert f"Would judge 5 run(s) for stamp {stamp} (source: grid)." in output
        assert run_ids == expected
        combined_run_ids.update(run_ids)

    assert len(combined_run_ids) == 20


def test_individual_run_scripts_partition_the_qwen3_sweep(tmp_path: Path) -> None:
    stamp = "20260711T220000Z"
    combined_run_ids: set[str] = set()

    for top_k, script in INDIVIDUAL_RUN_SCRIPTS.items():
        output = _run_script(
            script,
            tmp_path,
            RUN_STAMP=stamp,
        )
        run_ids = set(re.findall(r"--run-id ([^\s]+)", output))

        assert f"Sweep complete: 5 runs (stamp {stamp})" in output
        assert len(run_ids) == 5
        assert all(run_id.startswith("qwen3-14b-") for run_id in run_ids)
        assert all(f"-k{top_k}-" in run_id for run_id in run_ids)
        combined_run_ids.update(run_ids)

    assert len(combined_run_ids) == 20


def test_full_run_dry_run_omits_removed_comparison_steps(tmp_path: Path) -> None:
    output = _run_script("full_run.sh", tmp_path)
    removed_outputs = (
        "locomo-compare-pairs",
        "locomo-matrix-report",
        "pairing.json",
        "backend_overlap.json",
        "matrix_report.csv",
    )

    assert all(marker not in output for marker in removed_outputs)


def test_judge_grid_dry_run_emits_the_80_run_mem0_fact_matrix(tmp_path: Path) -> None:
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

    assert f"Would judge 80 run(s) for stamp {stamp} (source: grid)." in output
    assert _judge_run_ids(output) == _expected_full_run_ids(stamp)
    assert "prefix-gpu-jasper" not in output


def test_judge_discovery_accepts_current_and_supported_legacy_runs(tmp_path: Path) -> None:
    stamp = "20260710T120000Z"
    expected = {
        f"llama-kv-mem0-jasper10-k5-s0-{stamp}",
        f"llama-prefix-mem0-qdrant10-k5-s0-{stamp}",
        f"llama-kv-gpu-jasper10-k5-w0-{stamp}",
        f"llama-prefix-qdrant10-k5-w0-{stamp}",
    }
    # prefix-mem0-jasper left the sweep with the prompt-injection-jasper
    # baseline; discovery must no longer pick those runs up either.
    unsupported = (
        f"llama-prefix-gpu-jasper10-k5-{stamp}",
        f"llama-prefix-mem0-jasper10-k5-s0-{stamp}",
    )
    for run_id in (*expected, *unsupported):
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

    assert f"Would judge 4 run(s) for stamp {stamp} (source: discover)." in output
    assert _judge_run_ids(output) == expected
    assert all(run_id not in output for run_id in unsupported)


def test_full_run_cpu_store_dry_run_emits_the_64_run_kv_matrix(tmp_path: Path) -> None:
    stamp = "20260711T220000Z"
    output = _run_script("full_run_cpu_store.sh", tmp_path, RUN_STAMP=stamp)

    assert f"Sweep complete: 64 runs (stamp {stamp})" in output
    run_ids = re.findall(r"--run-id ([^\s]+)", output)
    assert len(run_ids) == 64
    assert set(run_ids) == {
        f"{model}-kvcpu-mem0-jasper10-k{top_k}-s{window}-{stamp}"
        for model in MODELS
        for top_k in TOP_KS
        for window in KV_WINDOWS
    }

    commands = [line for line in output.splitlines() if "--run-id" in line]
    assert all("--answer-backend vllm-kv" in command for command in commands)
    assert all("--kv-store-backend cpu" in command for command in commands)
    assert all("--kv-staging-slots 4" in command for command in commands)
    assert not any("vllm-prefix" in command for command in commands)
