#!/usr/bin/env bash
# Minimal reproduction: verify dataset + run one baseline + print summary.
#
# Usage (from repository root):
#   bash examples/quick_eval.sh
#   bash examples/quick_eval.sh --model gpt-4o-mini
#
# Prerequisites:
#   - .venv activated (or taco on PATH)
#   - taco data download
#   - configs/llm_config.yaml with a valid API key

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${1:-gpt-4o}"
if [[ "${1:-}" == "--model" ]]; then
  MODEL="${2:?Usage: $0 [--model MODEL]}"
fi

DATASET="beijing"
RESULTS="experiments/results/baseline_${MODEL//-/_}_taco_beijing.json"

echo "==> TACO-Benchmark quick eval"
echo "    Model  : $MODEL"
echo "    Dataset: $DATASET"
echo ""

echo "==> Verify dataset"
taco data verify

echo ""
echo "==> Run baseline"
taco eval run --model "$MODEL" --dataset "$DATASET" --output "$RESULTS"

echo ""
echo "==> Report"
taco eval report --pred "$RESULTS"

echo ""
echo "Done. Results saved to: $RESULTS"
