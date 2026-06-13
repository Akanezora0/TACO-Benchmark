# SQL Skeleton Generation

Generate diverse SQL skeletons (with JOINs, subqueries, aggregates) for each database in the benchmark.

## Overview

Combines CFG rules from legacy databases with expert skeleton examples to produce per-database SQL skeleton templates. Each database uses a deterministic seed derived from its name for reproducibility.

## Main script

| Script | Purpose |
|:--|:--|
| `generate_for_databases_improved.py` | Primary generator (recommended) |
| `generate_for_databases.py` | Legacy version (reference only) |

## Quick start

From the repository root:

```bash
python benchmark/generation/sql_skeleton_generation/generate_for_databases_improved.py
```

With custom sample counts:

```bash
python benchmark/generation/sql_skeleton_generation/generate_for_databases_improved.py \
  --num_samples 100 \
  --total_skeletons 200
```

## Key arguments

| Argument | Default | Description |
|:--|:--|:--|
| `--database_dir` | `benchmark/data/beijing/database` | Database schema directory |
| `--expert_file` | `benchmark/data/target/expert_skeletons_beijing.json` | Expert skeleton examples |
| `--output_dir` | `benchmark/data/beijing/output` | Output root |
| `--num_samples` | `100` | Structures per database |
| `--total_skeletons` | `200` | Skeletons per database |

## Output layout

Under `benchmark/data/beijing/output/`:

```text
output/
├── ast_cfg/         # CFG files per database
├── sql_structure/   # Parsed SQL structures
└── sql_skeleton/    # Final skeleton files used by sql_filling
```

## Design notes

- Expert examples include JOIN, subquery, and aggregate patterns
- Validation fixes common syntax issues before writing skeletons
- Per-database random seeds ensure diversity across databases while staying reproducible

## Next step

Pass skeletons to **SQL content filling**: see [../sql_filling/README.md](../sql_filling/README.md).
