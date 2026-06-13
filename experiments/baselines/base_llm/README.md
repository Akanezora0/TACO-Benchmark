# Base LLM Baseline Experiments

Direct Text-to-SQL evaluation with base LLMs (GPT-4o, DeepSeek-R1, etc.) using a standardized prompt and full-schema context.

## Files

| File | Purpose |
|:--|:--|
| `evaluate_baseline.py` | Main evaluation script (concurrent API calls) |
| `run_experiment.py` | Single experiment runner |
| `batch_evaluate.py` | Batch evaluation across models |
| `model_wrapper.py` | LLM client abstraction |
| `prompt_strategy.py` | Standardized baseline prompt template |
| `experiment_config.py` | Model configurations |

## Design principles

1. **Simple and direct** — no complex table-ranking heuristics
2. **Sufficient context** — include as many tables as the model context allows
3. **Direct Text-to-SQL** — single-shot SQL generation from NL + schema
4. **Concurrent API calls** — `ThreadPoolExecutor` for throughput

## Usage

### Evaluate on the official test split

```bash
python experiments/baselines/base_llm/evaluate_baseline.py \
  --model gpt-4o \
  --dataset taco_beijing
```

### Per-database evaluation (custom paths)

```bash
python experiments/baselines/base_llm/evaluate_baseline.py \
  --nl_query_dir benchmark/data/beijing/output/nl_query/Housing \
  --sql_dir benchmark/data/beijing/output/single/Housing \
  --db_path benchmark/data/beijing/database_chinese/Housing/Housing.db \
  --schema_file benchmark/data/beijing/database_chinese/Housing/Housing.json \
  --model gpt-4o \
  --output_file experiments/baselines/base_llm/results/beijing_Housing_gpt4o.json \
  --max_tables 100 \
  --max_columns_per_table 30 \
  --limit 100 \
  --max_workers 5
```

## Key parameters

| Parameter | Description |
|:--|:--|
| `--max_tables` | Max tables in prompt (GPT-4o: ~100–150; GPT-4 8K: ~20–30) |
| `--max_columns_per_table` | Columns per table (recommended 20–30) |
| `--limit` | Cap number of examples (for smoke tests) |
| `--max_workers` | API concurrency (default 5) |

## Model context windows

| Model | Context |
|:--|:--|
| GPT-4 | 8K |
| GPT-4o | 128K |
| GPT-o1 | 200K |
| DeepSeek-R1 | 64K |

## Output

JSON results with execution success rate, result-matching rate, per-query details, and configuration metadata.

Results are written under `experiments/baselines/base_llm/results/` (gitignored for large files).

## Related

- All baselines: [../README.md](../README.md)
- Evaluation metrics: [../../evaluation/README.md](../../evaluation/README.md)
