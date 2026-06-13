#!/usr/bin/env bash
# TACO-Benchmark — Linux / macOS / Git Bash bootstrap
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -n "${PYTHON_BIN:-}" ]; then
  PY="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "Python 3.10+ is required. Install from https://www.python.org/downloads/" >&2
  exit 1
fi

exec "$PY" scripts/setup_env.py "$@"
