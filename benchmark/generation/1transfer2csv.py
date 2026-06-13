#!/usr/bin/env python3
"""Deprecated shim — use ``preprocessing/transfer_to_csv.py`` instead."""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).parent / "preprocessing" / "transfer_to_csv.py"

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
