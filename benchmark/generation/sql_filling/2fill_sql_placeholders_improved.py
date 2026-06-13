#!/usr/bin/env python3
"""Deprecated shim — use ``fill_sql_placeholders.py`` instead."""

from __future__ import annotations

import runpy
from pathlib import Path

_TARGET = Path(__file__).with_name("fill_sql_placeholders.py")

if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
