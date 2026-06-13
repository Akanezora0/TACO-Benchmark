#!/usr/bin/env python3
"""Cross-platform environment bootstrap for TACO-Benchmark."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT / ".venv"
MIN_PYTHON = (3, 10)


def _step(msg: str) -> None:
    print(msg, flush=True)


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def _resolve_host_python() -> str:
    if os.environ.get("PYTHON_BIN"):
        return os.environ["PYTHON_BIN"]
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("Python not found. Install Python 3.10+.")


def _python_version_tuple(exe: str) -> tuple[int, int]:
    out = subprocess.check_output(
        [exe, "-c", "import sys; print(sys.version_info[:2])"],
        text=True,
    )
    parts = [int(x.strip()) for x in out.strip().strip("()").split(",")]
    return parts[0], parts[1]


def _venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _check_host_python() -> str:
    py = _resolve_host_python()
    major, minor = _python_version_tuple(py)
    if (major, minor) < MIN_PYTHON:
        need = ".".join(str(x) for x in MIN_PYTHON)
        raise RuntimeError(f"Python {need}+ required, got {major}.{minor}")
    return py


def _ensure_venv(host_python: str) -> Path:
    vpy = _venv_python()
    if not vpy.is_file():
        _step("[1/4] Create virtual environment (.venv)")
        _run([host_python, "-m", "venv", str(VENV_DIR)])
    else:
        _step("[1/4] Virtual environment exists (.venv)")
    if not vpy.is_file():
        raise RuntimeError("Failed to create .venv")
    return vpy


def _install_dependencies(vpy: Path) -> None:
    _step("[2/4] Upgrade pip")
    _run([str(vpy), "-m", "pip", "install", "--upgrade", "pip"])
    _step("[3/4] Install dependencies")
    _run([str(vpy), "-m", "pip", "install", "-r", "requirements.txt"])
    _run([str(vpy), "-m", "pip", "install", "-e", "."])


def _prepare_config() -> None:
    _step("[4/4] Prepare local config templates")
    pairs = [
        (
            ROOT / "benchmark/generation/sql_filling/config.yaml.example",
            ROOT / "benchmark/generation/sql_filling/config.yaml",
        ),
        (
            ROOT / "configs/llm_config.yaml.example",
            ROOT / "configs/llm_config.yaml",
        ),
        (
            ROOT / "configs/taco_sql_config.yaml.example",
            ROOT / "configs/taco_sql_config.yaml",
        ),
    ]
    for src, dst in pairs:
        if dst.is_file():
            continue
        if not src.is_file():
            _step(f"  skip (example missing): {src.relative_to(ROOT)}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _step(f"  + {dst.relative_to(ROOT)}")


def _print_next_steps() -> None:
    activate = (
        ".\\.venv\\Scripts\\Activate.ps1"
        if platform.system() == "Windows"
        else "source .venv/bin/activate"
    )
    _step("")
    _step("Environment is ready.")
    _step("Next steps:")
    _step(f"  1) Activate venv: {activate}")
    _step("  2) Configure LLM API:")
    _step("     configs/llm_config.yaml")
    _step("     benchmark/generation/sql_filling/config.yaml")
    _step("  3) Download dataset: taco data download")
    _step("  4) Verify dataset: taco data verify")
    _step("  5) Check install: taco info")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TACO-Benchmark setup")
    parser.parse_args(argv)

    os.chdir(ROOT)
    _step(f"TACO-Benchmark setup · {platform.system()} · {ROOT}")

    try:
        host_python = _check_host_python()
        vpy = _ensure_venv(host_python)
        _install_dependencies(vpy)
        _prepare_config()
        _print_next_steps()
        return 0
    except subprocess.CalledProcessError as exc:
        _step(f"\nCommand failed (exit {exc.returncode})")
        return exc.returncode or 1
    except RuntimeError as exc:
        _step(f"\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
