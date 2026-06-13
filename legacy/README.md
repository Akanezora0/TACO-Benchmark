# Legacy Code

This directory holds **superseded** generation scripts, maintenance utilities, and internal development notes that are no longer part of the recommended workflow.

## Contents

| Path | Description |
|:--|:--|
| `cross_database/archive/scripts/` | Old cross-database generation scripts (batch runners, tests, early fill logic) |
| `cross_database/archive/docs/` | Internal Chinese working notes from early development |
| `tools/cross_database/` | Dataset maintenance utilities (filtering, verification, statistics) |

### Maintenance tools (`tools/cross_database/`)

| Script | Purpose |
|:--|:--|
| `filter_best_sqls.py` | Filter cross-DB SQL by diversity and result quality |
| `verify_filtered_results.py` | Verify filtered SQL output |
| `check_target_progress.py` | Report progress vs. target counts |
| `statistics_all_datasets.py` | Count single- and cross-DB SQL across Beijing/US |
| `statistics_union_join_detailed.py` | Detailed JOIN/UNION statistics |

Shim: `benchmark/generation/cross_database/统计所有数据集.py` forwards to `statistics_all_datasets.py`.

## Usage

- Do **not** rely on these scripts for new work — use the canonical scripts under `benchmark/generation/`.
- Numbered filenames in `benchmark/generation/` remain as thin shims that forward to semantic names.
- Legacy scripts are kept for historical reference and may reference outdated paths.

## Related

- Active pipeline: [benchmark/generation/README.md](../benchmark/generation/README.md)
- Cross-DB (Beijing): [benchmark/generation/cross_database/README.md](../benchmark/generation/cross_database/README.md)
