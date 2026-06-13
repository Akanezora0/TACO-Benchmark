"""Experiment path helpers and subprocess runners."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .paths import BENCHMARK_DATA_DIR, EXPERIMENTS_DIR, PROJECT_ROOT

DATASET_ALIASES: dict[str, str] = {
    "beijing": "taco_beijing",
    "us": "taco_us",
    "taco-beijing": "taco_beijing",
    "taco-us": "taco_us",
}


def resolve_dataset(name: str) -> str:
    """Normalize dataset shorthand (e.g. ``beijing`` → ``taco_beijing``)."""
    key = name.strip().lower().replace("-", "_")
    return DATASET_ALIASES.get(key, name)


def default_test_data(dataset: str) -> Path:
    """Return the default evaluation split for a dataset."""
    resolved = resolve_dataset(dataset)
    return BENCHMARK_DATA_DIR / "final" / resolved / "test.json"


def default_baseline_output(model: str, dataset: str) -> Path:
    """Default path for baseline experiment results."""
    resolved = resolve_dataset(dataset)
    model_safe = model.replace("-", "_").replace(".", "_")
    return EXPERIMENTS_DIR / "results" / f"baseline_{model_safe}_{resolved}.json"


def default_ablation_output(setting: str, model: str, dataset: str) -> Path:
    """Default path for TACO-SQL ablation results."""
    resolved = resolve_dataset(dataset)
    model_safe = model.replace("-", "_").replace(".", "_")
    return EXPERIMENTS_DIR / "results" / f"{setting}_{model_safe}_{resolved}.json"


def ensure_project_on_path() -> None:
    """Allow importing top-level ``experiments`` package from the repo root."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def ensure_exists(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def run_python_script(
    script: Path | str,
    args: Sequence[str],
    *,
    cwd: Path | None = None,
) -> int:
    """Run a repository script with the current Python interpreter."""
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = PROJECT_ROOT / script_path
    cmd = [sys.executable, str(script_path), *args]
    result = subprocess.run(cmd, cwd=str(cwd or PROJECT_ROOT))
    return int(result.returncode or 0)


def load_results_json(path: Path) -> list[dict]:
    """Load per-query experiment results from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("per_query_results", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError(f"Unrecognized results format in {path}")


def summarize_results(results: list[dict]) -> dict:
    """Compute lightweight summary statistics from experiment results."""
    total = len(results)
    generated = sum(1 for r in results if r.get("generated_sql"))
    errors = sum(1 for r in results if r.get("generation_info", {}).get("error"))
    correct = sum(1 for r in results if r.get("is_correct"))
    summary: dict = {
        "total_queries": total,
        "with_generated_sql": generated,
        "generation_errors": errors,
    }
    if any("is_correct" in r for r in results):
        summary["correct_executions"] = correct
        summary["execution_accuracy"] = (correct / total) if total else 0.0
    return summary
