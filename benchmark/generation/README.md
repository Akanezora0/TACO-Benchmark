# Benchmark Data Generation

Module index for the data regeneration pipeline.

> **User guide:** [docs/GENERATION.md](../../docs/GENERATION.md) · **Install / API:** [docs/INSTALL.md](../../docs/INSTALL.md)

## Modules

| Module | README | Entry point |
|:--|:--|:--|
| SQL skeleton generation | [sql_skeleton_generation/README.md](sql_skeleton_generation/README.md) | `generate_for_databases_improved.py` |
| SQL content filling | [sql_filling/README.md](sql_filling/README.md) | `build_schema_graphs.py` → `fill_sql_placeholders.py` |
| NL query generation | [nl_query/README.md](nl_query/README.md) | `generate_nl_queries.py` |
| Cross-DB (Beijing) | [cross_database/README.md](cross_database/README.md) | `run_all.py` |
| Cross-DB (US) | [cross_database_us/README.md](cross_database_us/README.md) | `run_all.py` |
| Dataset preprocessing | [preprocessing/README.md](preprocessing/README.md) | one-off raw-data scripts |
| Single-DB end-to-end | — | `complete_pipeline_single_db.py` |
| Legacy / maintenance tools | [legacy/README.md](../../legacy/README.md) | archived scripts |

## Conventions

- **Canonical script names** use semantic filenames; numbered files are backward-compatible shims.
- **Intermediate artifacts** (`cross_db_graphs_join/`, etc.) are gitignored — see [GENERATION.md](../../docs/GENERATION.md#local-generation-artifacts-not-in-git).
