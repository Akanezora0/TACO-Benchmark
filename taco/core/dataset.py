"""Dataset download and verification utilities."""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path

from .paths import BENCHMARK_DATA_DIR, PROJECT_ROOT

GOOGLE_DRIVE_FILE_ID = "1Ynbv5eyEnM59mlEzqXZZsNh8sdHWH-nb"
GOOGLE_DRIVE_VIEW_URL = (
    "https://drive.google.com/file/d/1Ynbv5eyEnM59mlEzqXZZsNh8sdHWH-nb/view?usp=drive_link"
)
ARCHIVE_NAME = "taco-benchmark.tar.gz"

# Paths that must exist after a successful install (relative to benchmark/data).
REQUIRED_MARKERS: tuple[str, ...] = (
    "beijing/database_chinese",
    "beijing/output/single",
    "beijing/output/cross_db_final",
    "us/database",
    "us/output/single",
    "final/taco_beijing/test.json",
)


def archive_cache_path(cache_dir: Path | None = None) -> Path:
    root = cache_dir or (PROJECT_ROOT / ".cache")
    root.mkdir(parents=True, exist_ok=True)
    return root / ARCHIVE_NAME


def _ensure_gdown() -> None:
    try:
        import gdown  # noqa: F401
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown>=5.0.0"])


def download_archive(
    *,
    output: Path | None = None,
    force: bool = False,
) -> Path:
    """Download the TACO-Benchmark dataset archive from Google Drive."""
    _ensure_gdown()
    import gdown

    dest = output or archive_cache_path()
    if dest.is_file() and not force:
        return dest

    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    gdown.download(url, str(dest), quiet=False, fuzzy=True)
    if not dest.is_file():
        raise RuntimeError(f"Download failed; archive not found at {dest}")
    return dest


def _archive_top_level_dirs(archive_path: Path) -> set[str]:
    tops: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name or member.name == ".":
                continue
            tops.add(member.name.split("/")[0])
    return tops


def _relocate_extracted_tree(source: Path, target: Path) -> None:
    """Move *source* contents into *target*, replacing on conflict."""
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        dest = target / child.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(child), str(dest))


def extract_archive(
    archive_path: Path,
    *,
    data_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Extract the dataset archive into ``benchmark/data/``."""
    data_root = data_dir or BENCHMARK_DATA_DIR
    if data_root.exists() and any(data_root.iterdir()) and not force:
        missing = verify_dataset(data_root)
        if not missing:
            return data_root

    tmp_parent = PROJECT_ROOT / ".cache" / "taco_extract"
    if tmp_parent.exists():
        shutil.rmtree(tmp_parent)
    tmp_parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=tmp_parent)

    tops = {p.name for p in tmp_parent.iterdir()}
    data_root.mkdir(parents=True, exist_ok=True)

    if tops == {"benchmark"}:
        nested = tmp_parent / "benchmark" / "data"
        if nested.is_dir():
            _relocate_extracted_tree(nested, data_root)
        else:
            raise RuntimeError("Archive contains benchmark/ but no benchmark/data directory.")
    elif "beijing" in tops or "us" in tops or "final" in tops:
        _relocate_extracted_tree(tmp_parent, data_root)
    elif len(tops) == 1:
        only = tmp_parent / next(iter(tops))
        if only.is_dir() and {c.name for c in only.iterdir()} & {"beijing", "us", "final"}:
            _relocate_extracted_tree(only, data_root)
        else:
            _relocate_extracted_tree(tmp_parent, data_root)
    else:
        _relocate_extracted_tree(tmp_parent, data_root)

    shutil.rmtree(tmp_parent, ignore_errors=True)
    return data_root


def verify_dataset(data_dir: Path | None = None) -> list[str]:
    """Return a list of missing required paths (empty list means OK)."""
    root = data_dir or BENCHMARK_DATA_DIR
    missing: list[str] = []
    for rel in REQUIRED_MARKERS:
        path = root / rel
        if not path.exists():
            missing.append(rel)
    return missing


def download_and_extract(
    *,
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Download the archive (if needed) and extract into benchmark/data."""
    archive = download_archive(output=archive_cache_path(cache_dir), force=force)
    return extract_archive(archive, data_dir=data_dir, force=force)
