# TACO-Benchmark Dataset

## Download

| Source | Link |
|:--|:--|
| **Google Drive** | [TACO-Benchmark.zip](https://drive.google.com/file/d/1bPSYa8173XcFb1jqGQR5luNzCTxYmy_L/view?usp=sharing) |

### Automated download

```bash
# CLI (recommended)
taco data download

# Script
python scripts/download_dataset.py

# Verify layout
taco data verify
```

The archive is cached under `.cache/TACO-Benchmark.zip` and extracted to `benchmark/data/`.

### Manual download

1. Download `TACO-Benchmark.zip` from Google Drive.
2. Extract into `benchmark/data/`:

```bash
mkdir -p benchmark/data
unzip TACO-Benchmark.zip -d benchmark/data
```

If the archive contains a top-level `benchmark/data/` folder, extract from the repository root instead:

```bash
unzip TACO-Benchmark.zip -d .
```

3. Run `taco data verify` to confirm the layout.

---

## Overview

TACO contains **~14,500** Text-to-SQL examples across open-domain data lakes with three emphasis areas:

1. **Ambiguous natural language** — redundant context, implicit constraints, vague intent
2. **Unspecified target databases** — systems must retrieve relevant tables from large lakes
3. **Cross-database querying** — JOIN / UNION across 2–4 databases

### Subsets

| Subset | Databases | Single-DB SQL | Single-DB NL | Cross-DB SQL |
|:--|--:|--:|--:|--:|
| **TACO-Beijing** | 24 | 4,028 | 5,587 | 466 |
| **TACO-US** | 22 | 3,990 | 3,990 | 429 |
| **Total** | **46** | **8,018** | **9,577** | **895** |

Single-DB SQL and NL counts may differ within a subset when the two generation stages progress at different rates. For live counts after download:

```bash
python legacy/tools/cross_database/statistics_all_datasets.py
```

### Query-type distribution

Design distribution across the full benchmark:

| Type | ~Count | Share |
|:--|--:|--:|
| Single-database | ~11,700 | 80.5% |
| 2-database cross-DB | ~2,175 | 15.0% |
| 3-database cross-DB | ~638 | 4.4% |
| 4-database cross-DB | ~15 | 0.1% |

Cross-DB SQL in the released artifacts (895 total), split by database count:

| Cross-DB type | Beijing | US | Total |
|:--|--:|--:|--:|
| 2-database | ~375 | ~345 | ~720 |
| 3-database | ~82 | ~78 | ~160 |
| 4-database | ~9 | ~6 | ~15 |

---

## Directory layout

After extraction, `benchmark/data/` should look like:

```text
benchmark/data/
├── beijing/
│   ├── database_chinese/     # 24 SQLite DBs + JSON schemas
│   └── output/
│       ├── single/           # Per-database generated SQL + NL
│       ├── nl_query/
│       ├── cross_db_final/   # Cross-database SQL
│       ├── sql_skeleton/     # Intermediate artifacts
│       └── graph/            # Schema linking graphs
├── us/
│   ├── database/             # 22 US databases
│   └── output/               # Same structure as Beijing
└── final/                    # Evaluation splits
    └── taco_beijing/
        └── test.json
```

Experiment scripts default to `benchmark/data/final/{dataset}/test.json` (e.g. `taco_beijing`).

---

## Data formats

### Single-database example

```json
{
  "sql": "SELECT \"table_name\".\"column_name\" FROM \"table_name\" WHERE \"table_name\".\"column_name\" = 'value'",
  "sql_skeleton": "SELECT _ FROM _ WHERE _ = _",
  "natural_language_query": "Natural language description of the user query",
  "database": "Database name",
  "tables": {
    "table_name": ["column1", "column2"]
  },
  "metadata": {
    "has_join": false,
    "has_subquery": false,
    "has_aggregate": false
  },
  "cot_steps": {
    "step1_sql_analysis": "...",
    "step2_business_scenario": "...",
    "step3_user_scenario": "...",
    "step4_nl_generation": "..."
  }
}
```

### Cross-database example

```json
{
  "sql": "SELECT ... FROM \"db1\".\"table\" JOIN \"db2\".\"table\" ON ...",
  "sql_skeleton": "SELECT _ FROM _ JOIN _ ON _",
  "databases": ["database1", "database2"],
  "table_database_mapping": {
    "table_name": "database_name"
  },
  "results": [[...]],
  "metadata": {
    "num_databases": 2,
    "query_type": "JOIN"
  }
}
```

---

## Loading examples in Python

```python
import json
from pathlib import Path

# Single-database
single_dir = Path("benchmark/data/beijing/output/single")
for db_dir in single_dir.iterdir():
    for sql_file in db_dir.glob("generated_sql_*.json"):
        example = json.loads(sql_file.read_text(encoding="utf-8"))
        print(example["sql"], example.get("natural_language_query"))

# Cross-database
cross_dir = Path("benchmark/data/beijing/output/cross_db_final")
for sql_file in cross_dir.glob("cross_db_generated_sql_*.json"):
    example = json.loads(sql_file.read_text(encoding="utf-8"))
    print(example["sql"], example["databases"])

# Evaluation split
test = json.loads(
    Path("benchmark/data/final/taco_beijing/test.json").read_text(encoding="utf-8")
)
```

---

## Regenerating data

The benchmark can be regenerated with the pipeline under `benchmark/generation/`:

1. SQL skeleton generation
2. SQL content filling (LLM + schema graphs)
3. Natural-language query generation (CoT)

See [GENERATION.md](GENERATION.md) for the full pipeline. Cross-database schema graphs (`cross_db_graphs_join/`, ~10+ GB) are **not in git** — build them locally before fill scripts. Regeneration requires an OpenAI-compatible LLM API.
