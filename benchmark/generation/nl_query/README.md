# Natural Language Query Generation

Generate ambiguous, realistic natural-language queries from gold SQL using a Chain-of-Thought (CoT) pipeline.

## Overview

This stage turns executable SQL (from `sql_filling/`) into user-like NL questions that reflect TACO's ambiguity challenge: redundant context, implicit constraints, and vague intent.

## Main scripts

| Script | Purpose |
|:--|:--|
| `generate_nl_queries.py` | Primary NL generation (CoT prompts) |
| `complete_to_200.py` | Top up NL count per database to target |
| `regenerate_simple_queries.py` | Regenerate simpler NL variants |
| `build_template_library.py` | Build / expand NL template library |
| `analyze_real_data.py` | Analyze real user queries for template mining |
| `evaluate_models.py` | Compare NL quality across LLM backends |
| `cleanup_and_organize.py` | Organize generated NL files |

`4generate_nl_queries_improved.py` is a **backward-compatible shim** for `generate_nl_queries.py`.

`evaluate_baseline.py` is a **shim** — use `experiments/baselines/base_llm/evaluate_baseline.py` for baseline evaluation.

## Quick start

From the repository root:

```bash
python benchmark/generation/nl_query/generate_nl_queries.py \
  --sql_dir benchmark/data/beijing/output/single \
  --schema_dir benchmark/data/beijing/database_chinese \
  --output_dir benchmark/data/beijing/output/nl_query \
  --database Housing \
  --max_workers 5
```

## Prerequisites

- Filled SQL under `benchmark/data/.../output/single/`
- LLM API configured (`benchmark/generation/sql_filling/config.yaml`)
- Optional: `template_library.json` and `real_data_templates.json` for style diversity

## Output

Per-database NL JSON files under:

```text
benchmark/data/beijing/output/nl_query/{database}/
```

Each example links NL text to the corresponding SQL and may include `cot_steps` documenting the generation chain.

## End-to-end pipeline

For skeleton → SQL → NL in one flow, use:

```bash
python benchmark/generation/complete_pipeline_single_db.py --help
```

## Related

- Challenge examples: [../../../docs/EXAMPLES.md](../../../docs/EXAMPLES.md)
- SQL filling: [../sql_filling/README.md](../sql_filling/README.md)
