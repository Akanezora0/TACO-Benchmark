# TACO Experiments Framework

Baseline evaluations and TACO-SQL ablation studies on the TACO benchmark.

> **Run experiments:** [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) (CLI reference) · **Reproduce:** [examples/quick_eval.sh](../examples/quick_eval.sh)

## Directory structure

```text
experiments/
├── baselines/              # Baseline model experiments
│   ├── base_llm/          # GPT-4o, DeepSeek-R1, …
│   ├── llm_based/         # DIN-SQL, MAC-SQL
│   ├── sft_based/         # CodeS, Qwen2.5-Coder, …
│   └── hybrid/            # CHESS, Zero-NL2SQL, DIAL-SQL
├── taco_sql_exp/          # TACO-SQL ablations (origin → qr → qr_tl → qr_tl_qp)
├── evaluation/            # exec_eval, metrics, plots
├── results/               # Output JSON (gitignored large files)
└── docs/                  # Reviewer / paper notes
```

## Script entry points

### Baselines

```bash
python experiments/baselines/base_llm/run_experiment.py \
    --model gpt-4o \
    --test_data benchmark/data/final/taco_beijing/test.json \
    --output experiments/results/baseline_gpt_4o_taco_beijing.json
```

See [baselines/README.md](baselines/README.md) and [baselines/base_llm/README.md](baselines/base_llm/README.md).

### TACO-SQL ablation

```bash
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr_tl_qp \
    --model gpt-4o \
    --test_data benchmark/data/final/taco_beijing/test.json
```

See [taco_sql_exp/README.md](taco_sql_exp/README.md).

### Evaluation

```bash
python experiments/evaluation/exec_eval.py \
    --pred experiments/results/baseline_gpt_4o_taco_beijing.json \
    --gold benchmark/data/final/taco_beijing/test.json
```

See [evaluation/README.md](evaluation/README.md).

## Metrics

**Primary:** Execution Accuracy (EX) — share of queries where predicted SQL execution result matches gold.

```text
EX = (correct executions) / (total queries)
```

Additional: error breakdown, latency, token usage — see `evaluation/`.

## Results JSON shape

```json
{
    "model": "gpt-4o",
    "setting": "qr_tl_qp",
    "dataset": "taco_beijing",
    "total_queries": 1000,
    "execution_accuracy": 0.35,
    "per_query_results": [],
    "error_breakdown": {}
}
```

## Module documentation

| Module | README |
|:--|:--|
| Baselines | [baselines/README.md](baselines/README.md) |
| TACO-SQL | [taco_sql_exp/README.md](taco_sql_exp/README.md) |
| Evaluation | [evaluation/README.md](evaluation/README.md) |
| Reviewer notes | [docs/README.md](docs/README.md) |

## Fair comparison (summary)

- Same test split for all models
- Standardized prompts per setting (`baselines/base_llm/prompt_strategy.py`, `taco_sql_exp/prompts/`)
- Execution-based metric, not string match

Details: [docs/EXPERIMENT_FAIRNESS.md](docs/EXPERIMENT_FAIRNESS.md)
