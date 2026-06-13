#!/usr/bin/env python3
"""Deprecated shim — use ``check_generation_status.py`` instead."""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).with_name("check_generation_status.py")

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
