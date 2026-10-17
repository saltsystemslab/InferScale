"""Download a RAG dataset into its data directory before the preembed job runs."""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError
from benchmarks.rag import datasets as rag_datasets
from benchmarks.rag.config import RagBenchConfig

ARCHIVE_SUFFIXES = (".tgz", ".tar.gz")


def prepare_rag(data: dict[str, Any], runtime: RuntimeConfig) -> None:
    config = RagBenchConfig.from_dict(data, runtime=runtime)
    spec = rag_datasets.get_dataset(config.dataset_name)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in spec.download_urls.items():
        if filename.endswith(ARCHIVE_SUFFIXES):
            target = config.data_dir / spec.corpus_filename
            if not target.exists():
                _download_corpus_from_archive(url, target, config.data_dir)
        else:
            target = config.data_dir / filename
            if not target.exists():
                download(url, target)
    docs, queries = spec.load(config.data_dir)
    print(f"{spec.name}: {len(docs)} documents, {len(queries)} queries")


def _download_corpus_from_archive(url: str, target: Path, data_dir: Path) -> None:
    """Fetch an archive and keep only the corpus file, read without unpacking to disk."""
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        archive = Path(temporary) / "dataset.tgz"
        download(url, archive)
        with tarfile.open(archive) as bundle:
            member = next(
                (
                    entry
                    for entry in bundle.getmembers()
                    if Path(entry.name).name == target.name and entry.isfile()
                ),
                None,
            )
            if member is None:
                raise ConfigError(f"Dataset archive does not contain {target.name}.")
            contents = bundle.extractfile(member)
            if contents is None:
                raise ConfigError(f"Dataset archive entry {target.name} is not readable.")
            atomic_write_json(target, json.loads(contents.read()))


def download(url: str, target: Path) -> None:
    with urlopen(url, timeout=60) as response:
        payload = response.read()
    if target.suffix == ".json":
        atomic_write_json(target, json.loads(payload))
    else:
        target.write_bytes(payload)
