# Experiments Guide

How to run baseline evaluations and TACO-SQL ablation studies on TACO-Benchmark.

> **Detailed framework docs:** [experiments/README.md](../experiments/README.md)

---

## Prerequisites

1. Install and activate the environment — [INSTALL.md](INSTALL.md)
2. Download the dataset — `taco data download && taco data verify`
3. Configure LLM API — `configs/llm_config.yaml` (for API-based baselines)

---

## Quick start (CLI)

```bash
# Baseline on official test split (TACO-Beijing)
taco eval run --model gpt-4o --dataset beijing

# Summarize saved results
taco eval report --pred experiments/results/baseline_gpt_4o_taco_beijing.json

# TACO-SQL full setting (QR + Table Linking + Query Planning)
taco exp ablation --setting qr_tl_qp --model gpt-4o --dataset beijing

# Run all base-LLM baselines
taco exp run-all --dataset beijing --base-llm
```

Minimal reproduction script: [examples/quick_eval.sh](../examples/quick_eval.sh)

---

## CLI reference

### Dataset shorthand

| Input | Resolved |
|:--|:--|
| `beijing` | `taco_beijing` |
| `us` | `taco_us` |

Default test split: `benchmark/data/final/{dataset}/test.json`

### Evaluation (`taco eval`)

| Command | Description |
|:--|:--|
| `taco eval run` | Run a single base-LLM baseline on the test split |
| `taco eval batch` | Run multiple models in sequence |
| `taco eval report` | Summarize execution accuracy from a results JSON |
| `taco eval legacy-db` | Per-database evaluation with custom paths |

```bash
taco eval run --model gpt-4o --dataset beijing
taco eval run --model gpt-4o-mini --dataset us --test-data path/to/test.json
taco eval batch --models gpt-4o,gpt-4o-mini,deepseek-r1 --dataset beijing
taco eval report --pred experiments/results/baseline_gpt_4o_taco_beijing.json
taco eval legacy-db --database Housing --model gpt-4o --region beijing
```

### Experiments (`taco exp`)

| Command | Description |
|:--|:--|
| `taco exp baseline` | Alias for `taco eval run` |
| `taco exp ablation` | TACO-SQL ablation (origin / qr / qr_tl / qr_tl_qp) |
| `taco exp run-all` | Batch runner for baselines or ablations |

```bash
taco exp baseline --model gpt-4o --dataset beijing
taco exp ablation --setting qr_tl_qp --model gpt-4o --dataset beijing
taco exp run-all --dataset beijing --base-llm
taco exp run-all --dataset beijing --ablation --setting qr_tl
```

---

## Direct script invocation

The CLI wraps existing scripts under `experiments/`. You can call them directly:

```bash
# Base LLM baseline
python experiments/baselines/base_llm/run_experiment.py \
  --model gpt-4o \
  --test_data benchmark/data/final/taco_beijing/test.json \
  --output experiments/results/baseline_gpt_4o_taco_beijing.json

# TACO-SQL ablation
python experiments/taco_sql_exp/run_ablation.py \
  --setting qr_tl_qp \
  --model gpt-4o \
  --test_data benchmark/data/final/taco_beijing/test.json

# Execution accuracy
python experiments/evaluation/exec_eval.py \
  --pred experiments/results/baseline_gpt_4o_taco_beijing.json \
  --gold benchmark/data/final/taco_beijing/test.json
```

---

## Results layout

```text
experiments/results/
├── baseline_{model}_{dataset}.json    # Base-LLM results
└── {setting}_{model}_{dataset}.json   # Ablation results
```

Each result file contains per-query records with `generated_sql`, `is_correct`, and `generation_info`.

---

## Experimental settings

### Baselines

| Category | Methods |
|:--|:--|
| Base LLM | GPT-4o, GPT-4o-mini, GPT-o1, DeepSeek-R1, Llama3-70b, Qwen2-72b |
| LLM-based | DIN-SQL, MAC-SQL |
| SFT-based | CodeS-33B, Qwen2.5-Coder-32B, Deepseek-coder-6.7b |
| Hybrid | CHESS, Zero-NL2SQL, DIAL-SQL |

### TACO-SQL ablations

| Setting | Components |
|:--|:--|
| `origin` | Original query + full schema |
| `qr` | + Question Rewriting |
| `qr_tl` | + Question Rewriting + Table Linking |
| `qr_tl_qp` | Full TACO-SQL (+ Query Planning) |

Primary metric: **Execution Accuracy (EX)** — predicted SQL execution result matches gold.

---

## Configuration

| File | Purpose |
|:--|:--|
| `configs/llm_config.yaml` | Shared LLM settings for baselines |
| `configs/taco_sql_config.yaml` | TACO-SQL ablation components (copy from `.example`) |

---

## Related

- [INSTALL.md](INSTALL.md) — setup and troubleshooting
- [DATASET.md](DATASET.md) — test split format
- [EXAMPLES.md](EXAMPLES.md) — challenge examples
- [experiments/README.md](../experiments/README.md) — script entry points
- [experiments/docs/](../experiments/docs/) — reviewer / fairness notes
