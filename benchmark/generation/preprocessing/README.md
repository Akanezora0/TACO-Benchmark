# Dataset Preprocessing

One-off scripts for building the raw TACO dataset from source files (Excel/CSV → SQLite → schemas). **Not required** if you use the released dataset (`taco data download`).

## Scripts

| Script | Purpose |
|:--|:--|
| `transfer_to_csv.py` | Convert raw Excel/CSV to UTF-8 CSV |
| `parse_raw_table_name.py` | Parse table names and build mapping JSON |
| `create_database.py` | Create SQLite DBs from parsed CSV |
| `create_database_chinese.py` | Create DBs with Chinese table names |
| `create_database_from_existing.py` | Clone DBs with Chinese table name mappings |
| `extract_schema_chinese.py` | Export Chinese schemas from SQLite |
| `extract_new_schema.py` | Legacy schema extraction (mostly commented out) |
| `prepare_us_dataset.py` | Prepare US subset directory layout |

## Backward-compatible shims

Numbered entry points at `benchmark/generation/` root forward here:

- `1transfer2csv.py` → `preprocessing/transfer_to_csv.py`
- `2parse_raw_table_name.py` → `preprocessing/parse_raw_table_name.py`

## Typical order (from-scratch build)

```bash
python benchmark/generation/preprocessing/transfer_to_csv.py
python benchmark/generation/preprocessing/parse_raw_table_name.py
python benchmark/generation/preprocessing/create_database.py
python benchmark/generation/preprocessing/create_database_chinese.py
python benchmark/generation/preprocessing/extract_schema_chinese.py
```

Paths default to `benchmark/data/` relative to the repository root.

## Related

- Released dataset: [docs/DATASET.md](../../../docs/DATASET.md)
- Generation pipeline: [docs/GENERATION.md](../../../docs/GENERATION.md)
