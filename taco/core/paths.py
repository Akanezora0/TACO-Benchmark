"""Centralized path constants for TACO-Benchmark."""

from __future__ import annotations

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root (directory containing benchmark/ and experiments/)."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "benchmark").is_dir() and (candidate / "experiments").is_dir():
            return candidate

    # Fallback: taco/core/paths.py -> repo root is three levels up.
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = find_project_root()
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"
BENCHMARK_DATA_DIR = BENCHMARK_DIR / "data"
GENERATION_DIR = BENCHMARK_DIR / "generation"
SQL_FILLING_DIR = GENERATION_DIR / "sql_filling"
NL_QUERY_DIR = GENERATION_DIR / "nl_query"
CROSS_DATABASE_DIR = GENERATION_DIR / "cross_database"
CROSS_DATABASE_US_DIR = GENERATION_DIR / "cross_database_us"
SQL_SKELETON_DIR = GENERATION_DIR / "sql_skeleton_generation"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LEGACY_DIR = PROJECT_ROOT / "legacy"
