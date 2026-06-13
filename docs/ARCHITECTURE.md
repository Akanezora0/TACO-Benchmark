# Architecture

High-level map of the TACO-Benchmark repository: what each part does and where to make changes.

## System overview

```text
                    ┌─────────────────────────────────────┐
                    │           taco/ (CLI)               │
                    │  data · generate · eval · exp       │
                    └──────────────┬──────────────────────┘
                                   │ wraps
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   scripts/                  experiments/           benchmark/generation/
   download_dataset          baselines               sql_skeleton → sql_filling
   setup_env                 taco_sql_exp            → nl_query → cross_database
                             evaluation
```

**Data flow (benchmark construction):**

```text
raw sources ──► preprocessing/ ──► benchmark/data/
                      │
sql_skeleton_generation ──► sql_filling ──► nl_query ──► final JSON splits
                              │
                    cross_database (+ cross_database_us)
```

**Evaluation flow:**

```text
benchmark/data/final/{dataset}/test.json
        │
        ▼
experiments/baselines/ or taco_sql_exp/  ──► experiments/results/*.json
        │
        ▼
experiments/evaluation/exec_eval.py  ──► execution accuracy (EX)
```

## Directory boundaries

| Path | Role | Safe to refactor structure? |
|:--|:--|:--|
| `taco/` | Public CLI and shared helpers | Yes — keep command names stable |
| `benchmark/data/` | Released dataset (gitignored) | Don't change JSON field semantics |
| `benchmark/generation/` | Regeneration pipeline | Yes — prefer semantic script names + shims |
| `experiments/` | Baselines, ablations, metrics | Yes — don't change EX definition |
| `configs/` | Example templates only in git | Yes |
| `legacy/` | Archived scripts | Append-only; don't delete without notice |
| `docs/` | User-facing documentation | Yes |
| `examples/` | Minimal reproduction scripts | Yes |

## What not to commit

| Item | Reason |
|:--|:--|
| `benchmark/data/` | Download via `taco data download` |
| `cross_db_graphs_join/` and other generation artifacts | Regenerated locally (~10+ GB) |
| `configs/*.yaml` with real API keys | Use `*.example` templates |
| `__pycache__/`, `.venv/`, `*.log` | Build / runtime noise |

See `.gitignore` for the full list.

## Change guidelines

1. **Business logic** — SQL generation, filling, evaluation algorithms: change only when intentionally fixing behavior; document in commit message.
2. **Structure** — renaming, CLI wrappers, docs: welcome; keep numbered shims for backward compatibility until a major release.
3. **Experiments** — new baselines go under `experiments/baselines/{category}/`; share prompts via existing strategy modules.
4. **Documentation** — user workflows → `docs/`; per-script flags and paths → module `README.md`.

## Key entry points

| Task | Entry |
|:--|:--|
| Download dataset | `taco data download` |
| Run baseline | `taco eval run --model gpt-4o --dataset beijing` |
| Regenerate one DB | `taco generate single-db --database X --target-count N` |
| Cross-DB pipeline | `taco generate cross-db --region beijing` |
| Cross-DB status | `taco generate status --region beijing` |

## Related

- [CONTRIBUTING.md](../CONTRIBUTING.md)
- [GENERATION.md](GENERATION.md)
- [EXPERIMENTS.md](EXPERIMENTS.md)
