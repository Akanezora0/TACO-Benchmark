<div align="center">

# TACO-Benchmark

**A benchmark for open-domain Text-to-SQL with ambiguous and cross-database queries**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-4285F4?logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1Ynbv5eyEnM59mlEzqXZZsNh8sdHWH-nb/view?usp=drive_link)

<br/>

**[Dataset](#-dataset)** · **[Quick Start](#-quick-start)** · **[Examples](docs/EXAMPLES.md)** · **[Experiments](docs/EXPERIMENTS.md)**

</div>

**TACO** (Text-to-SQL with **A**mbiguous and **C**ross-database **O**pen-domain queries) evaluates Text-to-SQL systems on real-world data-lake scenarios. Unlike closed-domain benchmarks (Spider, BIRD), TACO requires handling vague user intent, unspecified target databases, and queries spanning multiple heterogeneous databases.

## Highlights

- **~14,500 examples** across finance, healthcare, transportation, housing, government, and more
- **Three core challenges** — ambiguous NL, open-domain table retrieval, cross-database SQL
- **Two regional subsets** — TACO-Beijing (24 DBs) and TACO-US (22 DBs)
- **Executable gold SQL** with validated execution results
- **Full experiment suite** — baselines, TACO-SQL ablations, execution-accuracy evaluation

## Dataset

The dataset is distributed separately (not in git):

| Source | File |
|:--|:--|
| [Google Drive](https://drive.google.com/file/d/1Ynbv5eyEnM59mlEzqXZZsNh8sdHWH-nb/view?usp=drive_link) | `taco-benchmark.tar.gz` |

### Statistics

| Subset | Databases | Single-DB SQL | Single-DB NL | Cross-DB SQL |
|:--|--:|--:|--:|--:|
| TACO-Beijing | 24 | 4,028 | 5,587 | 466 |
| TACO-US | 22 | — | 3,990 | — |

| Query type | Share |
|:--|--:|
| Single-database | ~80.5% |
| 2-database cross-DB | ~15.0% |
| 3-database cross-DB | ~4.4% |
| 4-database cross-DB | ~0.1% |

Details: **[docs/DATASET.md](docs/DATASET.md)**

## Quick Start

> Full guide: **[docs/INSTALL.md](docs/INSTALL.md)**

### 1. Install

```bash
git clone https://github.com/Akanezora0/TACO-Benchmark.git
cd TACO-Benchmark
python scripts/setup_env.py    # or: bash setup_env.sh
source .venv/bin/activate
```

### 2. Download data

```bash
taco data download
taco data verify
```

### 3. Configure LLM (for experiments / regeneration)

```bash
# Edit API settings (templates created by setup_env.py)
vim configs/llm_config.yaml
```

### 4. Load an example

```python
import json
from pathlib import Path

files = list(Path("benchmark/data/beijing/output/single").glob("*/*.json"))
example = json.loads(files[0].read_text(encoding="utf-8"))
print(example["natural_language_query"])
print(example["sql"])
```

### 5. Run a baseline experiment

```bash
taco eval run --model gpt-4o --dataset beijing
# or minimal repro:
bash examples/quick_eval.sh
```

See **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** for the CLI reference and **[experiments/README.md](experiments/README.md)** for the full framework.

## Three Core Challenges

| Challenge | What it tests |
|:--|:--|
| **Ambiguous NL** | Redundant context, implicit constraints, vague aggregation requests |
| **Unspecified databases** | Retrieve relevant tables from large heterogeneous lakes |
| **Cross-database SQL** | JOIN / UNION across 2–4 databases with weak relationships |

Representative NL/SQL pairs: **[docs/EXAMPLES.md](docs/EXAMPLES.md)**

## Project Layout

```text
TACO-Benchmark/
├── taco/                      # CLI (taco data · eval · exp)
├── benchmark/
│   ├── data/                  # Dataset — download via taco data (gitignored)
│   └── generation/            # Regeneration pipeline
│       ├── preprocessing/     # Raw data → SQLite (optional)
│       ├── sql_skeleton_generation/
│       ├── sql_filling/
│       ├── nl_query/
│       ├── cross_database/    # Beijing cross-DB (run_all.py)
│       └── cross_database_us/
├── experiments/
│   ├── baselines/             # Base LLM, DIN-SQL, CodeS, …
│   ├── taco_sql_exp/          # TACO-SQL ablations
│   ├── evaluation/            # Execution accuracy
│   └── docs/                  # Reviewer / paper notes
├── legacy/                    # Archived scripts + maintenance tools
├── docs/                      # User guides (see docs/README.md)
├── examples/                  # quick_eval.sh
├── configs/                   # *.example templates
└── scripts/                   # setup_env, download_dataset
```

Architecture details: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## CLI

```bash
taco --help
taco info

# Dataset
taco data download
taco data verify

# Data generation (requires LLM API)
taco generate single-db --database Housing --target-count 200 --region beijing
taco generate cross-db --region beijing
taco generate status --region beijing

# Baseline evaluation
taco eval run --model gpt-4o --dataset beijing
taco eval batch --models gpt-4o,gpt-4o-mini --dataset beijing
taco eval report --pred experiments/results/baseline_gpt_4o_taco_beijing.json

# TACO-SQL ablations & batch baselines
taco exp baseline --model gpt-4o --dataset beijing
taco exp ablation --setting qr_tl_qp --model gpt-4o --dataset beijing
taco exp run-all --dataset beijing --base-llm

# Legacy per-database evaluation (custom paths)
taco eval legacy-db --database Housing --model gpt-4o --region beijing
```

`--dataset` accepts shorthand (`beijing` → `taco_beijing`). Test split default: `benchmark/data/final/{dataset}/test.json`.

## Data Generation Pipeline

Benchmark data is produced in three stages:

1. **SQL skeleton generation** — CFG rules + expert examples
2. **SQL content filling** — LLM fills placeholders using schema-linking graphs
3. **NL query generation** — Chain-of-Thought NL from SQL

Scripts live under `benchmark/generation/`. Regeneration requires an LLM API.

Full guide: **[docs/GENERATION.md](docs/GENERATION.md)**

## Key Features

- Real-world query complexity (ambiguity, redundancy, implicit constraints)
- Open-domain table retrieval over multi-domain data lakes
- Cross-database JOIN and UNION (2–4 databases)
- Standardized baseline and ablation experiment framework
- Execution-accuracy (EX) as the primary metric

## Documentation

| Doc | Contents |
|:--|:--|
| [docs/README.md](docs/README.md) | Documentation index and layering guide |
| [docs/INSTALL.md](docs/INSTALL.md) | Environment setup, API config, troubleshooting |
| [docs/DATASET.md](docs/DATASET.md) | Download, layout, JSON formats |
| [docs/GENERATION.md](docs/GENERATION.md) | Data regeneration pipeline |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Baselines, ablations, CLI reference |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | Challenge examples with NL/SQL pairs |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Repo layout and change boundaries |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |
| [examples/README.md](examples/README.md) | Quick reproduction scripts |

## Requirements

- Python **3.10+**
- See `requirements.txt` (core) · `requirements-eval.txt` · `requirements-sft.txt`
- LLM API access for regeneration and API-based baselines

## Citation

If you use TACO-Benchmark in your research, please cite:

```bibtex
@misc{taco_benchmark,
  title  = {TACO: A Benchmark for Open-Domain Text-to-SQL with Ambiguous and Cross-Database Queries},
  author = {TACO-Benchmark Contributors},
  year   = {2026},
  url    = {https://github.com/Akanezora0/TACO-Benchmark}
}
```

## License

This project is released under the [MIT License](LICENSE).
