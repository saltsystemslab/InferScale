from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.common.clients import ChatResult
from benchmarks.common.config import RuntimeConfig
from benchmarks.common.vector_types import RetrievalMetrics
from benchmarks.memory.data import ConversationSample, QuestionAnswer
from benchmarks.memory.runtime_clients import RuntimeClients
from benchmarks.ttft_race import run as race
from memory_config import memory_runtime

Image = pytest.importorskip("PIL.Image")
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "mode, answer_backend, vector_backend, context_window",
    [("mem0", "prompt-injection", "qdrant", 0), ("inferscale", "kv-injection", "jasper", 50)],
)
def test_measure_runs_one_benchmark_cell_and_keeps_the_questions_after_warmup(
    monkeypatch: Any,
    tmp_path: Path,
    mode: str,
    answer_backend: str,
    vector_backend: str,
    context_window: int,
) -> None:
    questions = [f"Question {index}?" for index in range(race.WARMUP + 4)]
    sample = ConversationSample(
        sample_id="conv-1",
        turns=[],
        qa=[
            QuestionAnswer("conv-1", f"q{index}", question, f"gold {index}", "1")
            for index, question in enumerate(questions)
        ],
        raw={},
    )
    configs = []
    closed = []

    class FakeRetriever:
        memory = None

        def search(self, query: str, *, top_k: int) -> tuple[list[Any], RetrievalMetrics]:
            return [], RetrievalMetrics(
                embedding_time_ms=0.0, search_time_ms=0.0, total_time_ms=0.0
            )

        def close(self) -> None:
            pass

    class FakeMemoryBuilder:
        def __init__(self, config: Any) -> None:
            pass

        def load_fact_catalog(self, selected_sample: ConversationSample) -> tuple[Any, ...]:
            return ()

        def build_retriever_with_metrics(
            self, selected_sample: ConversationSample
        ) -> tuple[FakeRetriever, dict[str, Any]]:
            return FakeRetriever(), {}

        def log_embedding_cache_stats(self, memory: Any, sample_id: str) -> None:
            pass

    class FakeAnswerClient:
        def start_llm(self) -> None:
            pass

        def prepare_sample(self, selected_sample: ConversationSample) -> None:
            pass

        def close_sample(self) -> None:
            pass

        def close(self) -> None:
            closed.append(True)

        def answer_with_retrieved_memory(self, *, qa: QuestionAnswer, **_: Any) -> ChatResult:
            index = questions.index(qa.question)
            return ChatResult(
                content=f"answer {index}",
                ttft_ms=10.0 + index,
                metrics={"answer_generate_time_ms": 100.0 + index},
            )

    def build_clients(config: Any) -> RuntimeClients:
        configs.append(config)
        return RuntimeClients(answer_client=FakeAnswerClient(), judge_client=None)

    monkeypatch.setattr(
        "benchmarks.common.config.load_runtime_config",
        lambda: memory_runtime(storage={"runtime_root": str(tmp_path)}),
    )
    monkeypatch.setattr(RuntimeConfig, "apply_environment", lambda self: None)
    monkeypatch.setattr("benchmarks.memory.runner.build_clients", build_clients)
    monkeypatch.setattr("benchmarks.memory.runner.collect_system_metadata", lambda: {})
    monkeypatch.setattr(
        "benchmarks.memory.prediction.load_locomo", lambda *args, **kwargs: [sample]
    )
    monkeypatch.setattr("benchmarks.memory.prediction.SampleMemoryBuilder", FakeMemoryBuilder)
    out = tmp_path / "race" / f"{mode}.json"
    monkeypatch.setattr(
        sys, "argv", ["run", "measure", "--mode", mode, "--out", str(out), "--questions", "2"]
    )

    connector_log = logging.getLogger("inferscale.v1.kv.connector")
    info_enabled = connector_log.isEnabledFor(logging.INFO)

    race.main()

    # The connector logs every injection at INFO, inside the timed window of one side only.
    assert connector_log.isEnabledFor(logging.INFO) == info_enabled
    (config,) = configs
    assert (config.answer_backend, config.vector_backend) == (answer_backend, vector_backend)
    assert (config.top_k, config.context_window) == (100, context_window)
    assert (config.max_samples, config.max_questions, config.skip_judge) == (
        1,
        race.WARMUP + 2,
        True,
    )
    assert closed == [True]
    assert json.loads(out.read_text()) == {
        "mode": mode,
        "model": config.model,
        "top_k": 100,
        "context_window": context_window,
        "rows": [
            {
                "question": f"Question {index}?",
                "gold": f"gold {index}",
                "answer": f"answer {index}",
                "ttft_ms": 10.0 + index,
                "total_ms": 100.0 + index,
            }
            for index in (race.WARMUP, race.WARMUP + 1)
        ],
    }


def test_stream_spaces_words_evenly_between_the_first_token_and_the_end() -> None:
    row = {"answer": "one two three", "ttft_ms": 10.0, "total_ms": 30.0}
    assert race.stream(row) == [(10.0, "one "), (20.0, "two "), (30.0, "three")]
    assert race.stream({**row, "answer": "yes"}) == [(10.0, "yes")]


def _timelines(tmp_path: Path, **mem0_changes: Any) -> tuple[Path, Path]:
    """Three questions whose speedups are 2x, 8x, and 4x."""
    questions = ["First?", "Second?", "Third?"]

    def side(mode: str, window: int, timings: list[tuple[float, float]]) -> dict[str, Any]:
        return {
            "mode": mode,
            "model": "model",
            "top_k": 100,
            "context_window": window,
            "rows": [
                {
                    "question": question,
                    "gold": "gold",
                    "answer": "a short answer",
                    "ttft_ms": ttft,
                    "total_ms": total,
                }
                for question, (ttft, total) in zip(questions, timings)
            ],
        }

    fast = side("inferscale", 50, [(20.0, 120.0), (20.0, 100.0), (20.0, 60.0)])
    base = {**side("mem0", 0, [(40.0, 200.0), (160.0, 260.0), (80.0, 100.0)]), **mem0_changes}
    paths = tmp_path / "inferscale.json", tmp_path / "mem0.json"
    for path, timeline in zip(paths, (fast, base)):
        path.write_text(json.dumps(timeline))
    return paths


def test_gif_replays_the_median_speedup_question_in_the_harness_layout(
    monkeypatch: Any, tmp_path: Path, capsys: Any
) -> None:
    fast, base = _timelines(tmp_path)
    out = tmp_path / "race.gif"
    monkeypatch.setattr(sys, "argv", ["run", "gif", str(fast), str(base), "--out", str(out)])

    race.main()

    assert "question 2, first token 20 ms (InferScale) vs 80 ms (Mem0)" in capsys.readouterr().out
    gif = Image.open(out)
    assert gif.size == (960, 480)
    durations = []
    for index in range(gif.n_frames):
        gif.seek(index)
        durations.append(gif.info["duration"])
    # 100 ms replayed 20x slower at twenty frames per second, then the held final frame.
    assert durations == [50] * 40 + [3000]
    gif.seek(0)
    first = gif.convert("RGB")
    for point, color in (
        ((5, 5), (11, 18, 32)),
        ((29, 240), (56, 189, 248)),
        ((489, 240), (248, 113, 113)),
    ):
        assert max(abs(got - want) for got, want in zip(first.getpixel(point), color)) <= 8


@pytest.mark.parametrize(
    "mem0_changes, message",
    [
        ({"mode": "inferscale"}, "InferScale timeline first"),
        ({"rows": []}, "different questions"),
    ],
)
def test_gif_refuses_swapped_or_mismatched_timelines(
    monkeypatch: Any, tmp_path: Path, mem0_changes: dict[str, Any], message: str
) -> None:
    fast, base = _timelines(tmp_path, **mem0_changes)
    monkeypatch.setattr(
        sys, "argv", ["run", "gif", str(fast), str(base), "--out", str(tmp_path / "race.gif")]
    )

    with pytest.raises(SystemExit, match=message):
        race.main()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository skeleton whose configured interpreter prints its arguments."""
    target = tmp_path / "repo $(printf keep-literal)"
    shutil.copytree(ROOT / "scripts", target / "scripts")
    for name in (
        "benchmarks/__init__.py",
        "benchmarks/common/__init__.py",
        "benchmarks/common/config.py",
        "benchmarks/common/paths.py",
        "benchmarks/common/environment.py",
        "benchmarks/ttft_race/run.sh",
    ):
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target / name)
    interpreter = target / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n"
    )
    interpreter.chmod(0o755)
    (target / "configs").mkdir()
    (target / "configs/runtime.json").write_text(
        json.dumps(
            {
                "storage": {"runtime_root": str(target / "data")},
                "build": {"venv_dir": str(interpreter.parents[1])},
            }
        )
    )
    return target


def _launch(repo: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "BENCHMARK_ENV_FILE": str(repo / "absent.env")}
    return subprocess.run(
        ["/bin/bash", str(repo / "benchmarks/ttft_race/run.sh"), *args],
        cwd=repo.parent,
        env=env,
        text=True,
        capture_output=True,
    )


def test_launcher_measures_each_side_in_its_own_process_then_draws_the_gif(repo: Path) -> None:
    process = _launch(repo)

    assert process.returncode == 0, process.stderr
    out = repo / "data/results/ttft-race"
    module = ["-m", "benchmarks.ttft_race.run"]
    assert [json.loads(line) for line in process.stdout.splitlines()] == [
        [*module, "measure", "--mode", "mem0", "--out", str(out / "mem0.json")],
        [*module, "measure", "--mode", "inferscale", "--out", str(out / "inferscale.json")],
        [
            *module,
            "gif",
            str(out / "inferscale.json"),
            str(out / "mem0.json"),
            "--out",
            str(out / "ttft_race.gif"),
        ],
    ]


def test_launcher_rejects_arguments_before_any_work(repo: Path) -> None:
    process = _launch(repo, "--questions", "16")

    assert process.returncode == 2
    assert "takes no arguments" in process.stderr
    assert process.stdout == ""
