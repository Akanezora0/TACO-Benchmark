#!/usr/bin/env python3
"""Deprecated shim — use ``experiments/baselines/base_llm/evaluate_baseline.py`` instead."""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[3] / "experiments/baselines/base_llm/evaluate_baseline.py"

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
