from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from loguru import logger

from benchmarks.common.atomic_files import atomic_write_json
from benchmarks.common.config import RuntimeConfig
from benchmarks.common.json_fields import ConfigError
from benchmarks.rag import datasets as rag_datasets
from benchmarks.rag.config import RagBenchConfig

ARCHIVE_SUFFIXES = (".tgz", ".tar.gz")
DOWNLOAD_TIMEOUT_S = 60


def prepare_rag(run: dict[str, Any], runtime: RuntimeConfig) -> None:
    config = RagBenchConfig.from_dict(run, runtime=runtime)
    dataset = rag_datasets.get_dataset(config.dataset_name)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in dataset.download_urls.items():
        is_archive = filename.endswith(ARCHIVE_SUFFIXES)
        target = config.data_dir / (dataset.corpus_filename if is_archive else filename)
        if target.exists():
            continue
        if is_archive:
            _download_corpus_from_archive(url, target)
        else:
            download(url, target)
    documents, queries = dataset.load(config.data_dir)
    logger.info("{}: {} documents, {} queries", dataset.name, len(documents), len(queries))


def _download_corpus_from_archive(url: str, corpus_path: Path) -> None:
    with tempfile.TemporaryDirectory(dir=corpus_path.parent) as temporary_folder:
        archive_path = Path(temporary_folder) / "dataset.tgz"
        download(url, archive_path)
        with tarfile.open(archive_path) as archive:
            corpus = archive.extractfile(_file_in_archive(archive, corpus_path.name))
            if corpus is None:
                raise ConfigError(f"Dataset archive entry {corpus_path.name} is not readable.")
            atomic_write_json(corpus_path, json.loads(corpus.read()))


def _file_in_archive(archive: tarfile.TarFile, filename: str) -> tarfile.TarInfo:
    for member in archive.getmembers():
        if member.isfile() and Path(member.name).name == filename:
            return member
    raise ConfigError(f"Dataset archive does not contain {filename}.")


def download(url: str, target: Path) -> None:
    with urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
        payload = response.read()
    if target.suffix == ".json":
        atomic_write_json(target, json.loads(payload))
    else:
        target.write_bytes(payload)
