from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from benchmarks.common import launcher
from benchmarks.common.config import ConfigError, RuntimeConfig
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

    runs = launcher.expand_runs(plan, runtime, "run", "stamp")

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
        "benchmark": "rag", "models": ["llama", "qwen"],
        "sweeps": {"small": {"top_k": [1, 3]}},
    }
    runs = launcher.expand_runs(plan_for(tmp_path, data, sweep="small"), runtime, "run", "stamp")
    assert len(runs) == 4
    assert {run["top_k"] for run in runs} == {1, 3}
    assert all(run["answer_backend"] == "kv-injection" for run in runs)
    assert all(run["skip_judge"] for run in runs)
    assert len({run["run_id"] for run in runs}) == 4


def test_empty_rag_sweep_uses_authored_run_defaults(tmp_path, runtime):
    data = {"benchmark": "rag", "top_k": 3, "sweeps": {"single": {}}}
    runs = launcher.expand_runs(plan_for(tmp_path, data, sweep="single"), runtime, "run", "stamp")
    assert len(runs) == 1
    assert runs[0]["top_k"] == 3


@pytest.mark.parametrize("settings", [
    {"model": "llama", "models": ["qwen"]},
    {"models": ["llama", "llama"]},
    {"model": False}, {"model": 0}, {"model": ""}, {"run_id": ""},
])
def test_rag_launcher_preserves_schema_validation(tmp_path, runtime, settings):
    plan = plan_for(tmp_path, {"benchmark": "rag", **settings})
    with pytest.raises(ConfigError):
        launcher.expand_runs(plan, runtime, "run", "stamp")


def test_fixed_rag_run_id_is_preserved_for_one_cell_and_rejected_for_many(tmp_path, runtime):
    data = {"benchmark": "rag", "run_id": "existing"}
    plan = plan_for(tmp_path, data)
    assert launcher.expand_runs(plan, runtime, "run", "stamp")[0]["run_id"] == "existing"
    data["models"] = ["llama", "qwen"]
    with pytest.raises(ConfigError, match="duplicate run IDs"):
        launcher.expand_runs(plan_for(tmp_path, data), runtime, "run", "stamp")


def test_rag_judge_requires_existing_run_id(tmp_path, runtime):
    plan = plan_for(tmp_path, {"benchmark": "rag"})
    with pytest.raises(ConfigError, match="existing run_id"):
        launcher.expand_runs(plan, runtime, "judge", "stamp")


def test_rag_prepare_dry_run_prints_resolved_defaults(tmp_path, runtime, monkeypatch, capsys):
    source = tmp_path / "rag.json"
    source.write_text(json.dumps({"benchmark": "rag", "run_id": "prepare"}))
    manifest = {"family": "rag", "run_configs": [str(source)]}
    monkeypatch.setattr(launcher, "prepare_rag", lambda *_: pytest.fail("dry run downloaded data"))
    monkeypatch.setattr(launcher, "run_job", lambda *_: pytest.fail("dry run started a process"))
    assert launcher.execute_manifest(manifest, tmp_path / "manifest.json", runtime, "prepare", dry_run=True) == 0
    assert f"Prepare dataset multihoprag in {tmp_path / 'data/multihoprag'}" in capsys.readouterr().out


def test_rag_archive_preparation_reads_only_expected_json_and_reuses_it(tmp_path, runtime, monkeypatch):
    from benchmarks.rag import datasets

    loaded = []
    spec = SimpleNamespace(
        name="qasper", corpus_filename="qasper.json",
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
            for filename, payload in (("nested/qasper.json", b'{"valid": true}'), ("../outside.json", b"{}")):
                entry = tarfile.TarInfo(filename)
                entry.size = len(payload)
                archive.addfile(entry, io.BytesIO(payload))

    monkeypatch.setattr(launcher, "_download", download)
    data = {"benchmark": "rag", "dataset": "qasper"}
    launcher.prepare_rag(data, runtime)
    launcher.prepare_rag(data, runtime)

    assert downloads == ["http://dataset/archive.tgz"]
    assert loaded == [{"valid": True}, {"valid": True}]
    assert list((tmp_path / "data/qasper").iterdir()) == [tmp_path / "data/qasper/qasper.json"]
    assert not (tmp_path / "data/outside.json").exists()
