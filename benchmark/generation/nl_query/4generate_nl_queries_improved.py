#!/usr/bin/env python3
"""Deprecated shim — use ``generate_nl_queries.py`` instead."""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).with_name("generate_nl_queries.py")

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
