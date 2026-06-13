# Cross-Database SQL Generation (US)

Cross-database JOIN SQL generation for the TACO-US subset. Mirrors the Beijing pipeline with English schemas and US database layouts.

## Target distribution

| Type | Target share | Target count |
|:--|--:|--:|
| Single-database | 80.5% | 4,830 |
| 2-database cross-DB | 15.0% | 900 |
| 3-database cross-DB | 4.4% | 264 |
| 4-database cross-DB | 0.1% | 6 |

## Scripts

| Script | Purpose |
|:--|:--|
| `check_status.py` | Check generation progress |
| `analyze_joinable_tables.py` | Analyze joinable table pairs |
| `run_all.py` | Run the full pipeline (or a single step) |

Steps 2–5 call shared scripts in `../cross_database/` via `resolve_script()`.

## Quick start

From the repository root:

```bash
cd benchmark/generation/cross_database_us

# Check status
python check_status.py

# Full pipeline
python run_all.py

# Single step
python run_all.py --step 1
```

### Step by step

```bash
python analyze_joinable_tables.py
python run_all.py --from-step 2    # skeletons → graphs → SQL
```

## Output

```text
benchmark/data/us/output/
├── cross_db_single_join/           # Generated SQL
└── cross_db_single_join_backup/    # Backed-up valid SQL
```

## Configuration

LLM API: `benchmark/generation/sql_filling/config.yaml`.

See [../cross_database/README.md](../cross_database/README.md) for shared technical details (join scoring, ATTACH DATABASE, iterative cleanup).
