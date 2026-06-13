# Cross-Database SQL Generation (Beijing)

Generate cross-database JOIN SQL for the TACO-Beijing subset (2–4 databases per query).

## Pipeline

```text
analyze joinable table pairs
        ↓
generate cross-DB SQL skeletons
        ↓
build schema-linking graphs
        ↓
LLM fill placeholders → executable SQL
        ↓
cleanup / backup / statistics
```

## Core scripts

| Stage | Script | Purpose |
|:--|:--|:--|
| Analysis | `analyze_joinable_tables.py` | Find joinable table pairs across databases |
| Skeletons | `generate_cross_db_skeletons_join.py` | Build JOIN skeletons (2/3/4 DB) |
| Graphs | `cross_db_1build_schema_graphs.py` | Schema graphs for cross-DB skeletons |
| Fill (2 DB) | `generate_more_join_sqls.py` | Batch-generate 2-database SQL |
| Fill (3/4 DB) | `generate_3db_4db_sqls.py` | Generate 3- and 4-database SQL |
| Cleanup | `cleanup_failed_sqls.py` | Remove failed 2-DB SQL |
| Cleanup | `cleanup_failed_3db_4db_sqls.py` | Remove failed 3/4-DB SQL |
| Backup | `backup_new_results.py` | Backup SQL with valid execution results |
| Stats | `statistics_join_sqls.py` | Summarize generation progress |
| Status | `check_generation_status.py` | Check remaining targets |
| Organize | `cleanup_and_organize.py` | Archive obsolete scripts and tidy the directory |
| Runner | `run_all.py` | One-click pipeline (steps 1–5) |

Numbered filenames (e.g. `2generate_cross_db_skeletons_join.py`) remain as **backward-compatible shims** that forward to the scripts above.

Maintenance utilities (`filter_best_sqls.py`, etc.) moved to [`legacy/tools/cross_database/`](../../../legacy/tools/cross_database/).

## Quick start

From the repository root:

```bash
cd benchmark/generation/cross_database

# Check status
python check_generation_status.py

# Full pipeline (interactive)
python run_all.py

# Non-interactive
python run_all.py -y

# Single step
python run_all.py --step 3
```

### Manual step-by-step
python analyze_joinable_tables.py
python generate_cross_db_skeletons_join.py \
  --num_skeletons_2db 500 --num_skeletons_3db 150 --num_skeletons_4db 20
python cross_db_1build_schema_graphs.py
python generate_more_join_sqls.py
python generate_3db_4db_sqls.py
python backup_new_results.py
```

### Iterative refinement

```bash
python cleanup_failed_sqls.py
python cleanup_failed_3db_4db_sqls.py
python backup_new_results.py
python generate_more_join_sqls.py
python generate_3db_4db_sqls.py
```

## Data paths

| Path | In git? | Description |
|:--|:--|:--|
| `database_combinations.json` | Yes | Database combination plan (small config) |
| `joinable_table_pairs.json` | No | Joinable table pairs (large; gitignored) |
| `cross_db_skeletons_join.json` | No | Cross-DB SQL skeletons (regenerate locally) |
| `cross_db_graphs_join/` | No | Schema graph files (regenerate locally; ~10+ GB) |
| `skeletons/` | No | Per-DB skeleton shards (regenerate locally) |
| `benchmark/data/beijing/output/cross_db_final/` | No | Final cross-DB SQL output (under dataset) |

> **Fresh clone:** run `analyze_joinable_tables.py` → `generate_cross_db_skeletons_join.py` → `cross_db_1build_schema_graphs.py` before fill/cleanup scripts.

## Configuration

LLM settings: `benchmark/generation/sql_filling/config.yaml` (copy from `config.yaml.example`).

Target counts are defined inside `generate_more_join_sqls.py` and `generate_3db_4db_sqls.py`.

## Technical notes

- Joinable pairs are scored by column-name similarity (ID, code, name keywords)
- Multi-DB SQL uses SQLite `ATTACH DATABASE`
- 2-DB queries have higher execution success than 3-DB; iterative cleanup + regeneration is expected

## Legacy archive

Superseded scripts and internal docs live under [`legacy/cross_database/archive/`](../../../legacy/cross_database/archive/) and are not part of the recommended workflow.
