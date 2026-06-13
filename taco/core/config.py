"""Configuration loading helpers for TACO-Benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_DIR, SQL_FILLING_DIR

DEFAULT_LLM_CONFIG: dict[str, Any] = {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.1,
    "max_tokens": 8000,
    "max_workers": 4,
    "api_url": "https://api.openai.com/v1",
    "api_key": "your-api-key-here",
}


def _apply_env_overrides(llm: dict[str, Any]) -> dict[str, Any]:
    merged = dict(llm)
    if os.environ.get("TACO_API_KEY"):
        merged["api_key"] = os.environ["TACO_API_KEY"]
    if os.environ.get("TACO_API_URL"):
        merged["api_url"] = os.environ["TACO_API_URL"]
    if os.environ.get("TACO_MODEL"):
        merged["model"] = os.environ["TACO_MODEL"]
    return merged


def load_llm_config(config_file: Path | str | None = None) -> dict[str, Any]:
    """
    Load LLM settings from a YAML file.

    Search order when *config_file* is None:
    1. ``benchmark/generation/sql_filling/config.yaml``
    2. ``configs/llm_config.yaml``
    """
    if config_file is not None:
        candidates = [Path(config_file)]
    else:
        candidates = [
            SQL_FILLING_DIR / "config.yaml",
            CONFIGS_DIR / "llm_config.yaml",
        ]

    for path in candidates:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        llm = data.get("llm", data)
        if isinstance(llm, dict):
            return _apply_env_overrides({**DEFAULT_LLM_CONFIG, **llm})

    return _apply_env_overrides(dict(DEFAULT_LLM_CONFIG))


def ensure_local_config_from_example(
    dst: Path,
    example: Path,
) -> Path:
    """Copy *example* to *dst* when the destination file does not exist."""
    if dst.is_file():
        return dst
    if not example.is_file():
        raise FileNotFoundError(f"Config example not found: {example}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(example.read_bytes())
    return dst
