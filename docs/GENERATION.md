# Data Generation Pipeline

This guide describes how to **regenerate** the TACO-Benchmark dataset. The released dataset can be downloaded directly — regeneration is optional and requires an OpenAI-compatible LLM API.

> **Quick path:** Most users only need `taco data download`. See [DATASET.md](DATASET.md).

---

## Overview

```text
sql_skeleton_generation/     CFG + expert examples → SQL skeleton templates
        ↓
sql_filling/                 Schema-linking graphs + LLM → executable SQL
        ↓
nl_query/                    Chain-of-Thought NL from gold SQL
        ↓
cross_database/              Cross-DB JOIN SQL (Beijing, 2–4 databases)
cross_database_us/           Cross-DB JOIN SQL (US subset)
```

| Stage | Input | Output |
|:--|:--|:--|
| Skeleton generation | Database schemas, expert examples | `output/sql_skeleton/` |
| Schema graphs | Skeletons + schemas | `output/graph/` (`.graphml`, metadata JSON) |
| SQL filling | Skeletons + graphs + schemas | `output/single/{db}/generated_sql_*.json` |
| NL generation | Filled SQL + schemas | `output/nl_query/{db}/` |
| Cross-DB SQL | Joinable table pairs | `output/cross_db_final/` |

Module-level details:

| Module | README |
|:--|:--|
| SQL skeleton generation | [sql_skeleton_generation/README.md](../benchmark/generation/sql_skeleton_generation/README.md) |
| SQL content filling | [sql_filling/README.md](../benchmark/generation/sql_filling/README.md) |
| NL query generation | [nl_query/README.md](../benchmark/generation/nl_query/README.md) |
| Cross-DB (Beijing) | [cross_database/README.md](../benchmark/generation/cross_database/README.md) |
| Cross-DB (US) | [cross_database_us/README.md](../benchmark/generation/cross_database_us/README.md) |
| Preprocessing | [preprocessing/README.md](../benchmark/generation/preprocessing/README.md) |

Module index: [benchmark/generation/README.md](../benchmark/generation/README.md).

## Prerequisites

1. **Environment** — see [INSTALL.md](INSTALL.md)
2. **Dataset schemas** — `taco data download` (SQLite DBs and schema files under `benchmark/data/`)
3. **LLM API** — copy and edit:

```bash
cp benchmark/generation/sql_filling/config.yaml.example \
   benchmark/generation/sql_filling/config.yaml
# Set api_key, api_url, model
```

Environment overrides: `TACO_API_KEY`, `TACO_API_URL`, `TACO_MODEL`.

### CLI shortcuts

```bash
# Single database (skeleton → graph → SQL → NL)
taco generate single-db --database Housing --target-count 200 --region beijing

# Cross-database pipeline
taco generate cross-db --region beijing
taco generate cross-db --region beijing --step 3 --yes

# Check cross-DB progress
taco generate status --region beijing
```

---

## End-to-end: single database

The shortcut script runs skeleton → graph → SQL fill → NL for one database:

```bash
python benchmark/generation/complete_pipeline_single_db.py \
  --database Housing \
  --target_count 200 \
  --output_dir benchmark/data/beijing/output
```

Useful flags:

| Flag | Purpose |
|:--|:--|
| `--skip_skeleton` | Reuse existing skeletons |
| `--skip_graph` | Reuse existing graphs |
| `--skip_fill` | Skip SQL generation |
| `--skip_nl` | Skip NL generation |
| `--schema_dir` | Override schema directory |
| `--database_dir` | Override SQLite database directory |

Strategy: generate ~3× skeletons, fill SQL until the target count is reached, then generate NL queries in one pass.

---

## Step-by-step (Beijing single-DB)

Run from the repository root.

### 1. SQL skeletons

```bash
python benchmark/generation/sql_skeleton_generation/generate_for_databases_improved.py \
  --database_dir benchmark/data/beijing/database \
  --output_dir benchmark/data/beijing/output \
  --num_samples 100 \
  --total_skeletons 200
```

### 2. Schema-linking graphs

```bash
python benchmark/generation/sql_filling/build_schema_graphs.py \
  --skeleton_dir benchmark/data/beijing/output/sql_skeleton \
  --database_dir benchmark/data/beijing/database_chinese \
  --output_dir benchmark/data/beijing/output/graph
```

### 3. SQL content filling

```bash
python benchmark/generation/sql_filling/fill_sql_placeholders.py \
  --skeleton_dir benchmark/data/beijing/output/sql_skeleton \
  --database_dir benchmark/data/beijing/database_chinese \
  --graph_dir benchmark/data/beijing/output/graph \
  --output_dir benchmark/data/beijing/output \
  --max_retries 3
```

### 4. Natural-language queries

```bash
python benchmark/generation/nl_query/generate_nl_queries.py \
  --sql_dir benchmark/data/beijing/output/single \
  --schema_dir benchmark/data/beijing/database_chinese \
  --output_dir benchmark/data/beijing/output/nl_query \
  --database Housing \
  --max_workers 5
```

---

## Cross-database SQL (Beijing)

```bash
cd benchmark/generation/cross_database

python run_all.py              # Beijing: steps 1-5
# or manually:
python analyze_joinable_tables.py
python generate_cross_db_skeletons_join.py \
  --num_skeletons_2db 500 --num_skeletons_3db 150 --num_skeletons_4db 20
python cross_db_1build_schema_graphs.py
python generate_more_join_sqls.py      # 2-database SQL
python generate_3db_4db_sqls.py        # 3- and 4-database SQL
python backup_new_results.py
```

Iterative refinement (cleanup failed SQL → backup → regenerate) is expected for 3- and 4-database queries. See [cross_database/README.md](../benchmark/generation/cross_database/README.md).

### Local generation artifacts (not in git)

Cross-database intermediate files are **regenerated locally** and are listed in `.gitignore`:

| Path | Produced by |
|:--|:--|
| `cross_db_skeletons_join.json` | `generate_cross_db_skeletons_join.py` |
| `cross_db_graphs_join/` | `cross_db_1build_schema_graphs.py` |
| `skeletons/` | Per-DB skeleton shards during generation |
| `joinable_table_pairs.json` | `analyze_joinable_tables.py` |

After a fresh clone, run the cross-database steps in order through `cross_db_1build_schema_graphs.py` before SQL fill scripts. The graph step may take significant time and disk space (~10+ GB for a full Beijing run).

---

## Cross-database SQL (US)

```bash
cd benchmark/generation/cross_database_us

python check_status.py
python analyze_joinable_tables.py
python run_all.py              # or: python run_all.py --step 1
```

Shared skeleton/graph/fill scripts live in `../cross_database/`. See [cross_database_us/README.md](../benchmark/generation/cross_database_us/README.md).

---

## Script naming

Canonical scripts use **semantic names** (e.g. `build_schema_graphs.py`, `fill_sql_placeholders.py`). Numbered filenames from earlier development (`1build_schema_graphs_improved.py`, `2fill_sql_placeholders_improved.py`, …) remain as thin **backward-compatible shims**.

Superseded scripts are archived under [`legacy/`](../legacy/README.md).

---

## Output format

Single-DB SQL JSON (simplified):

```json
{
  "sql": "SELECT \"table\".\"col\" FROM \"table\" WHERE \"table\".\"col\" = 'value'",
  "results": [[...]],
  "sql_skeleton": "SELECT _ FROM _ WHERE _ = _",
  "database": "DatabaseName",
  "tables": { "table": ["table.col1", "table.col2"] },
  "metadata": {
    "has_join": false,
    "has_subquery": false,
    "has_aggregate": false
  }
}
```

NL examples add `natural_language_query` and optional `cot_steps`. Full field reference: [DATASET.md](DATASET.md).

---

## Cost and runtime notes

- Regenerating the full benchmark is **LLM-intensive** (thousands of API calls).
- Cross-database generation uses iterative cleanup; 3-DB and 4-DB SQL have lower success rates than 2-DB.
- Use `--max_workers` where supported to parallelize, but respect API rate limits.
- For a smoke test, run `complete_pipeline_single_db.py` on one database with a small `--target_count` (e.g. 5).

---

## Related

- [INSTALL.md](INSTALL.md) — environment and API setup
- [DATASET.md](DATASET.md) — download, layout, evaluation split
- [EXAMPLES.md](EXAMPLES.md) — challenge examples (ambiguity, cross-DB)
- [ARCHITECTURE.md](ARCHITECTURE.md) — repository layout
