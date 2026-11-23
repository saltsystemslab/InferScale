from __future__ import annotations

import io
import json
import tarfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.common.config import ConfigError, RuntimeConfig
from benchmarks.common.launcher import execute, plans, rag_prepare
from benchmarks.rag.config import RagBenchConfig


@pytest.fixture
def runtime(tmp_path):
    data = {"models": {"llama": "configured/llama", "qwen": "configured/qwen"}}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(data))
    return RuntimeConfig.from_dict(data, root=tmp_path, path=path)


def plan_for(tmp_path, data, *, sweep=None):
    source = tmp_path / "rag.json"
    source.write_text(json.dumps(data))
    plan = {"family": "rag", "configs": [str(source)]}
    if sweep is not None:
        plan["sweep"] = sweep
    return plan


@pytest.mark.parametrize(
    "settings",
    [{}, {"model": "test/model"}, {"models": ["llama"]}, {"run_id": None}, {"model": None}],
)
def test_minimal_rag_json_expands_using_loader_defaults(tmp_path, runtime, settings):
    data = {"benchmark": "rag", "inferscale": {}, **settings}
    original = deepcopy(data)
    expected = RagBenchConfig.from_dict(data, runtime=runtime)
    plan = plan_for(tmp_path, data)

    runs = plans.expand_runs(plan, runtime, "run", "stamp")

    assert len(runs) == 1
    actual = RagBenchConfig.from_dict(runs[0], runtime=runtime)
    assert actual.model == expected.model
    assert actual.top_k == expected.top_k == 15
    assert actual.answer_backend == expected.answer_backend == "kv-injection"
    assert actual.context_window == expected.context_window == 5
    assert actual.dataset_name == expected.dataset_name == "multihoprag"
    assert actual.run_id.endswith("-kv-k15-s5-stamp")
    assert json.loads(Path(plan["configs"][0]).read_text()) == original


def test_partial_rag_sweep_uses_config_defaults_for_other_axes(tmp_path, runtime):
    data = {
        "benchmark": "rag",
        "models": ["llama", "qwen"],
        "sweeps": {"small": {"top_k": [1, 3]}},
    }
    runs = plans.expand_runs(plan_for(tmp_path, data, sweep="small"), runtime, "run", "stamp")
    assert len(runs) == 4
    assert {run["top_k"] for run in runs} == {1, 3}
    assert all(run["answer_backend"] == "kv-injection" for run in runs)
    assert all(run["skip_judge"] for run in runs)
    assert len({run["run_id"] for run in runs}) == 4


def test_rag_sweep_uses_prompt_injection_run_marker(tmp_path, runtime):
    data = {
        "benchmark": "rag",
        "model": "llama",
        "sweeps": {"standard": {"top_k": [1, 3], "answer_backends": ["prompt-injection"]}},
    }
    runs = plans.expand_runs(plan_for(tmp_path, data, sweep="standard"), runtime, "run", "stamp")

    assert {run["run_id"] for run in runs} == {
        "multihoprag-llama-prompt-injection-k1-s5-stamp",
        "multihoprag-llama-prompt-injection-k3-s5-stamp",
    }
    assert all(run["answer_backend"] == "prompt-injection" for run in runs)


def test_rag_sweep_uses_lmcache_blend_run_marker(tmp_path, runtime):
    data = {
        "benchmark": "rag",
        "model": "llama",
        "lmcache": {"max_local_cpu_size_gb": 8},
        "sweeps": {"lmcache": {"top_k": [15], "answer_backends": ["lmcache-blend"]}},
    }
    runs = plans.expand_runs(plan_for(tmp_path, data, sweep="lmcache"), runtime, "run", "stamp")

    assert [run["run_id"] for run in runs] == ["multihoprag-llama-lmcache-blend-k15-s5-stamp"]
    assert runs[0]["answer_backend"] == "lmcache-blend"
    assert runs[0]["lmcache"] == {"max_local_cpu_size_gb": 8}
    del data["lmcache"]
    with pytest.raises(ConfigError, match="rag.lmcache is required"):
        plans.expand_runs(plan_for(tmp_path, data, sweep="lmcache"), runtime, "run", "stamp")


def test_empty_rag_sweep_uses_authored_run_defaults(tmp_path, runtime):
    data = {"benchmark": "rag", "top_k": 3, "sweeps": {"single": {}}}
    runs = plans.expand_runs(plan_for(tmp_path, data, sweep="single"), runtime, "run", "stamp")
    assert len(runs) == 1
    assert runs[0]["top_k"] == 3


@pytest.mark.parametrize(
    "settings",
    [
        {"model": "llama", "models": ["qwen"]},
        {"models": ["llama", "llama"]},
        {"model": False},
        {"model": 0},
        {"model": ""},
        {"run_id": ""},
    ],
)
def test_rag_launcher_preserves_schema_validation(tmp_path, runtime, settings):
    plan = plan_for(tmp_path, {"benchmark": "rag", **settings})
    with pytest.raises(ConfigError):
        plans.expand_runs(plan, runtime, "run", "stamp")


def test_fixed_rag_run_id_is_preserved_for_one_cell_and_rejected_for_many(tmp_path, runtime):
    data = {"benchmark": "rag", "run_id": "existing"}
    plan = plan_for(tmp_path, data)
    assert plans.expand_runs(plan, runtime, "run", "stamp")[0]["run_id"] == "existing"
    data["models"] = ["llama", "qwen"]
    with pytest.raises(ConfigError, match="duplicate run IDs"):
        plans.expand_runs(plan_for(tmp_path, data), runtime, "run", "stamp")


def test_rag_judge_requires_existing_run_id(tmp_path, runtime):
    plan = plan_for(tmp_path, {"benchmark": "rag"})
    with pytest.raises(ConfigError, match="existing run_id"):
        plans.expand_runs(plan, runtime, "judge", "stamp")


def test_rag_prepare_dry_run_logs_resolved_defaults(tmp_path, runtime, monkeypatch, launcher_log):
    source = tmp_path / "rag.json"
    source.write_text(json.dumps({"benchmark": "rag", "run_id": "prepare"}))
    manifest = {"family": "rag", "run_configs": [str(source)]}
    monkeypatch.setattr(execute, "prepare_rag", lambda *_: pytest.fail("dry run downloaded data"))
    monkeypatch.setattr(execute, "run_job", lambda *_: pytest.fail("dry run started a process"))
    assert (
        execute.execute_manifest(
            manifest, tmp_path / "manifest.json", runtime, "prepare", dry_run=True
        )
        == 0
    )
    assert launcher_log == [
        "[1/1] prepare",
        f"Prepare dataset multihoprag in {tmp_path / 'data/multihoprag'}",
        f"Launch manifest: {tmp_path / 'manifest.json'}",
    ]


def test_rag_prepare_downloads_the_dataset_without_starting_a_job(tmp_path, runtime, monkeypatch):
    source = tmp_path / "rag.json"
    source.write_text(json.dumps({"benchmark": "rag", "run_id": "prepare"}))
    manifest = {"family": "rag", "run_configs": [str(source)]}
    prepared = []
    monkeypatch.setattr(execute, "prepare_rag", lambda data, _: prepared.append(data["run_id"]))
    monkeypatch.setattr(execute, "run_job", lambda *_: pytest.fail("prepare started a process"))

    assert (
        execute.execute_manifest(
            manifest, tmp_path / "manifest.json", runtime, "prepare", dry_run=False
        )
        == 0
    )

    assert prepared == ["prepare"]


def test_rag_run_starts_the_run_job_without_a_catalog_check(tmp_path, runtime, monkeypatch):
    source = tmp_path / "rag.json"
    source.write_text(json.dumps({"benchmark": "rag", "run_id": "answers"}))
    manifest = {"family": "rag", "run_configs": [str(source)]}
    started = []

    def run_job(job, log_file):
        started.append((job, log_file))
        return 0

    monkeypatch.setattr(execute, "run_job", run_job)

    assert (
        execute.execute_manifest(
            manifest, tmp_path / "manifest.json", runtime, "run", dry_run=False
        )
        == 0
    )

    assert started == [
        (
            {
                "family": "rag",
                "config": str(source),
                "runtime_config": str(runtime.path),
                "stage": "run",
            },
            tmp_path / "logs" / "answers.run.log",
        )
    ]


def test_rag_archive_preparation_reads_only_expected_json_and_reuses_it(
    tmp_path, runtime, monkeypatch, launcher_log
):
    from benchmarks.rag import datasets

    loaded = []
    spec = SimpleNamespace(
        name="qasper",
        corpus_filename="qasper.json",
        download_urls={"archive.tgz": "http://dataset/archive.tgz"},
    )

    def load(path):
        loaded.append(json.loads((path / spec.corpus_filename).read_text()))
        return ["document"], ["query"]

    spec.load = load
    monkeypatch.setattr(datasets, "get_dataset", lambda _: spec)
    downloads = []

    def download(url, target):
        downloads.append(url)
        with tarfile.open(target, "w:gz") as archive:
            for filename, payload in (
                ("nested/qasper.json", b'{"valid": true}'),
                ("../outside.json", b"{}"),
            ):
                entry = tarfile.TarInfo(filename)
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))

    monkeypatch.setattr(rag_prepare, "download", download)
    data = {"benchmark": "rag", "dataset": "qasper"}
    rag_prepare.prepare_rag(data, runtime)
    rag_prepare.prepare_rag(data, runtime)

    assert downloads == ["http://dataset/archive.tgz"]
    assert loaded == [{"valid": True}, {"valid": True}]
    assert list((tmp_path / "data/qasper").iterdir()) == [tmp_path / "data/qasper/qasper.json"]
    assert not (tmp_path / "data/outside.json").exists()
    assert launcher_log == ["qasper: 1 documents, 1 queries"] * 2


def test_rag_preparation_downloads_only_the_files_that_are_missing(
    tmp_path, runtime, monkeypatch, launcher_log
):
    from benchmarks.rag import datasets

    spec = SimpleNamespace(
        name="multihoprag",
        corpus_filename="corpus.json",
        download_urls={
            "corpus.json": "http://dataset/corpus.json",
            "queries.json": "http://dataset/queries.json",
        },
        load=lambda path: (["document"], ["query", "query"]),
    )
    monkeypatch.setattr(datasets, "get_dataset", lambda _: spec)
    data_dir = tmp_path / "data/multihoprag"
    data_dir.mkdir(parents=True)
    (data_dir / "queries.json").write_text("{}")
    downloads = []

    def download(url, target):
        downloads.append((url, target))
        target.write_text("{}")

    monkeypatch.setattr(rag_prepare, "download", download)
    rag_prepare.prepare_rag({"benchmark": "rag"}, runtime)
    rag_prepare.prepare_rag({"benchmark": "rag"}, runtime)

    assert downloads == [("http://dataset/corpus.json", data_dir / "corpus.json")]
    assert launcher_log == ["multihoprag: 1 documents, 2 queries"] * 2


def test_rag_archive_without_the_corpus_file_is_rejected(tmp_path, runtime, monkeypatch):
    from benchmarks.rag import datasets

    spec = SimpleNamespace(
        name="qasper",
        corpus_filename="qasper.json",
        download_urls={"archive.tar.gz": "http://dataset/archive.tar.gz"},
        load=lambda path: pytest.fail("loaded an incomplete dataset"),
    )
    monkeypatch.setattr(datasets, "get_dataset", lambda _: spec)

    def download(url, target):
        with tarfile.open(target, "w:gz") as archive:
            archive.addfile(tarfile.TarInfo("other.json"), io.BytesIO(b""))
            folder = tarfile.TarInfo("qasper.json")
            folder.type = tarfile.DIRTYPE
            archive.addfile(folder)

    monkeypatch.setattr(rag_prepare, "download", download)

    with pytest.raises(ConfigError, match="Dataset archive does not contain qasper.json"):
        rag_prepare.prepare_rag({"benchmark": "rag", "dataset": "qasper"}, runtime)

    assert list((tmp_path / "data/qasper").iterdir()) == []


def test_download_rewrites_json_and_keeps_other_files_verbatim(tmp_path, monkeypatch):
    payloads = {
        "http://dataset/corpus.json": b'{"b": 1,\n "a": [2]}',
        "http://dataset/archive.tgz": b"\x1f\x8braw bytes",
        "http://dataset/broken.json": b"not json",
    }
    timeouts = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return self.payload

    def urlopen(url, timeout):
        timeouts.append(timeout)
        return Response(payloads[url])

    monkeypatch.setattr(rag_prepare, "urlopen", urlopen)

    rag_prepare.download("http://dataset/corpus.json", tmp_path / "corpus.json")
    rag_prepare.download("http://dataset/archive.tgz", tmp_path / "archive.tgz")
    with pytest.raises(json.JSONDecodeError):
        rag_prepare.download("http://dataset/broken.json", tmp_path / "broken.json")

    assert (tmp_path / "corpus.json").read_text() == '{"a":[2],"b":1}\n'
    assert (tmp_path / "archive.tgz").read_bytes() == b"\x1f\x8braw bytes"
    assert not (tmp_path / "broken.json").exists()
    assert timeouts == [60, 60, 60]
