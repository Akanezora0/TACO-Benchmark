# SQL Content Filling

Fill SQL skeletons with concrete table names, column names, and values using schema-linking graphs and an LLM.

## Overview

This module:

1. Builds **SQL–Schema Linking Graphs** from skeletons and database schemas
2. Extracts relevant tables/columns from graphs to build compact prompts
3. Fills skeleton placeholders and validates SQL by execution against SQLite

Key improvements over the initial implementation:

- Uses graph connectivity and foreign-key edges for JOIN skeletons
- Enriches prompts with table/column metadata and FK relationships
- Falls back to metadata or random selection when graph extraction fails

## Files

```text
sql_filling/
├── build_schema_graphs.py          # Step 1: build schema graphs (canonical)
├── fill_sql_placeholders.py        # Step 2: fill placeholders (main)
├── build_schema_graphs_optimized.py
├── graph_extractor.py              # Graph-based prompt compression
├── config.yaml.example             # LLM config template → copy to config.yaml
└── README.md
```

Numbered filenames (`1build_schema_graphs_improved.py`, `2fill_sql_placeholders_improved.py`, …) are **backward-compatible shims**.

## Prerequisites

- Dataset downloaded (`taco data download`)
- SQL skeletons under `benchmark/data/{beijing|us}/output/sql_skeleton/`
- SQLite databases under `benchmark/data/{beijing|us}/database_chinese/` (or `database/` for US)
- LLM API configured in `config.yaml` (see `config.yaml.example`)

## Usage

Run from the repository root:

### Step 1 — Build schema graphs

Use **`build_schema_graphs.py`** (recommended) — full GraphML + metadata JSON.

Use **`build_schema_graphs_optimized.py`** only when disk space matters: it writes compact JSON metadata without GraphML files. The main fill pipeline (`fill_sql_placeholders.py`) expects the standard graph layout from `build_schema_graphs.py`.

```bash
python benchmark/generation/sql_filling/build_schema_graphs.py \
  --skeleton_dir benchmark/data/beijing/output/sql_skeleton \
  --database_dir benchmark/data/beijing/database_chinese \
  --output_dir benchmark/data/beijing/output/graph
```

Outputs per database:

- `{db}/{db}_graph_{i}.graphml`
- `{db}/{db}_metadata_{i}.json`

### Step 2 — Fill SQL placeholders

```bash
python benchmark/generation/sql_filling/fill_sql_placeholders.py \
  --skeleton_dir benchmark/data/beijing/output/sql_skeleton \
  --database_dir benchmark/data/beijing/database_chinese \
  --graph_dir benchmark/data/beijing/output/graph \
  --output_dir benchmark/data/beijing/output \
  --max_retries 3
```

Outputs:

- `benchmark/data/beijing/output/single/{database}/generated_sql_{i}.json`

## Configuration

Copy and edit:

```bash
cp benchmark/generation/sql_filling/config.yaml.example \
   benchmark/generation/sql_filling/config.yaml
```

```yaml
llm:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 8000
  api_url: "https://api.openai.com/v1"
  api_key: "your-api-key-here"
```

Or set environment variables: `TACO_API_KEY`, `TACO_API_URL`, `TACO_MODEL`.

## Output format

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

## Notes

- Run Step 1 before Step 2 (graphs are required for best results)
- Failed generations are retried up to `--max_retries` times
- For end-to-end single-database generation, see `benchmark/generation/complete_pipeline_single_db.py`
