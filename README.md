<div align="center">

# TACO-Benchmark: A benchmark for open-domain Text-to-SQL with ambiguous and cross-database queries

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Akanezora/TACO-Benchmark)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-4285F4?logo=google-drive&logoColor=white)](https://drive.google.com/file/d/1Ynbv5eyEnM59mlEzqXZZsNh8sdHWH-nb/view?usp=drive_link)

<br/>

**[News](#-news)** · **[Why TACO?](#-why-taco)** · **[Dataset](#-dataset)** · **[Examples](#-representative-examples)** · **[Quick Start](#-quick-start)** · **[Docs](docs/README.md)**

<br/>

⭐️ If you find TACO helpful, a **star** on GitHub would mean a lot to us — thanks for your support!

</div>

## 📢 News

| Date | Update |
|:--|:--|
| **2026** | 🎉 TACO has been **accepted to VLDB 2026**! Paper and camera-ready details coming soon. |

---

**TACO** (Text-to-SQL with **A**mbiguous and **C**ross-database **O**pen-domain queries) is a benchmark for **real-world data-lake Text-to-SQL**. Unlike Spider or BIRD — where the target database is known and schemas are clean — TACO asks models to cope with **vague user questions**, **unspecified databases**, and **queries spanning multiple heterogeneous databases**.

## ✨ Why TACO?

Existing benchmarks largely assume a single, well-defined schema. In practice, users query **open data lakes** with messy intent and weak cross-source relationships. TACO fills this gap with three stress tests:

| Challenge | What makes it hard |
|:--|:--|
| 💬 **Ambiguous NL** | Redundant context, implicit constraints, and vague terms that do not map 1:1 to SQL |
| 🔍 **Open-domain retrieval** | The target database is not given — systems must find relevant tables across domains |
| 🔗 **Cross-database SQL** | A single question may require JOIN or UNION across 2–4 databases with weak keys |

**What you get**

- 📦 **~14,500** high-quality Text-to-SQL instances with **executable gold SQL** and validated results
- 🗄️ **46 databases** across finance, healthcare, transportation, housing, government, and more
- 🌏 **Two regional subsets** — TACO-Beijing (24 DBs) and TACO-US (22 DBs)
- 📏 **Standard evaluation splits** and a baseline / ablation suite (execution accuracy as the primary metric)

## 📊 Dataset

The dataset is not included in git. Download from either source:

| Source | Link |
|:--|:--|
| **Hugging Face** (recommended) | [Akanezora/TACO-Benchmark](https://huggingface.co/datasets/Akanezora/TACO-Benchmark) |
| **Google Drive** | [`taco-benchmark.tar.gz`](https://drive.google.com/file/d/1Ynbv5eyEnM59mlEzqXZZsNh8sdHWH-nb/view?usp=drive_link) |

```python
# Hugging Face
from huggingface_hub import snapshot_download
snapshot_download("Akanezora/TACO-Benchmark", repo_type="dataset", local_dir="./taco-benchmark")
```

```bash
# Google Drive (via project CLI)
taco data download && taco data verify
```

### Scale by subset

| Subset | Databases | Single-DB SQL | Single-DB NL | Cross-DB SQL |
|:--|--:|--:|--:|--:|
| TACO-Beijing | 24 | 4,028 | 5,587 | 466 |
| TACO-US | 22 | 3,990 | 3,990 | 429 |
| **Total** | **46** | **8,018** | **9,577** | **895** |

### Query-type distribution

| Type | Share | ~Count |
|:--|--:|--:|
| Single-database | 80.5% | ~11,700 |
| 2-database cross-DB | 15.0% | ~2,175 |
| 3-database cross-DB | 4.4% | ~638 |
| 4-database cross-DB | 0.1% | ~15 |

Format details and directory layout: **[docs/DATASET.md](docs/DATASET.md)**

## 💡 Representative examples

### 1 · Ambiguous natural language

The user asks for summaries, comparisons, and deviation analysis — but the gold SQL is a focused filter on one year:

> *"I need to verify the government fund budget revenue for the entire year of 2018. Please help me summarize the budget revenue and actual received amounts for all government fund budget projects in 2018… find projects with large deviations… This data is very important for annual settlement review…"*

```sql
SELECT "finance_bureau_budget_execution_report"."ProjectName",
       "finance_bureau_budget_execution_report"."BudgetRevenue2018",
       "finance_bureau_budget_execution_report"."Year"
FROM "finance_bureau_budget_execution_report"
WHERE "finance_bureau_budget_execution_report"."Year" = '2018'
```

**Takeaway:** models must extract the core intent from noisy, redundant NL — not over-generate SQL for every phrase the user mentions.

### 2 · Cross-database JOIN

A single question spans two databases with an implicit join key:

> *"Query the construction project completion filing information from the Housing database, join with the public toilet information from the Life Services database, count the number of public toilets for each project name, and sort by the number of public toilets in descending order."*

```sql
SELECT "Housing"."construction_project_completion_filing"."ProjectName",
       COUNT("LifeServices"."public_toilet_info"."SequenceNumber") AS "ToiletCount"
FROM "Housing"."construction_project_completion_filing"
JOIN "LifeServices"."public_toilet_info"
  ON "Housing"."construction_project_completion_filing"."SequenceNumber"
   = "LifeServices"."public_toilet_info"."SequenceNumber"
GROUP BY "Housing"."construction_project_completion_filing"."ProjectName"
ORDER BY "ToiletCount" DESC
```

**Takeaway:** systems must plan across databases, align schemas, and produce valid multi-DB SQL (e.g., `ATTACH DATABASE` in SQLite).

More examples (open-domain retrieval, UNION, 3–4 DB queries): **[docs/EXAMPLES.md](docs/EXAMPLES.md)**

## 🚀 Quick Start

```bash
git clone https://github.com/Akanezora0/TACO-Benchmark.git && cd TACO-Benchmark
python scripts/setup_env.py && source .venv/bin/activate
# Download data from Hugging Face (see Dataset section) or: taco data download && taco data verify
taco eval run --model gpt-4o --dataset beijing   # baseline evaluation
```

Setup, API configuration, and troubleshooting: **[docs/INSTALL.md](docs/INSTALL.md)** · Minimal repro script: **[examples/quick_eval.sh](examples/quick_eval.sh)**

## 📈 Evaluation

We provide baseline experiments (GPT-4o, DIN-SQL, CodeS, …), TACO-SQL ablations, and execution-accuracy (EX) evaluation. See **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** and **[experiments/README.md](experiments/README.md)**.

## 📚 Documentation

| Doc | Contents |
|:--|:--|
| [docs/DATASET.md](docs/DATASET.md) | Download, formats, directory layout |
| [docs/EXAMPLES.md](docs/EXAMPLES.md) | Full challenge examples with NL/SQL pairs |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Baselines, ablations, CLI reference |
| [docs/INSTALL.md](docs/INSTALL.md) | Environment setup |
| [docs/GENERATION.md](docs/GENERATION.md) | Data regeneration pipeline (optional) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Repository layout |

<!-- ## 📝 Citation

```bibtex
@misc{taco_benchmark,
  title  = {TACO: A Benchmark for Open-Domain Text-to-SQL with Ambiguous and Cross-Database Queries},
  author = {TACO-Benchmark Contributors},
  year   = {2026},
  url    = {https://github.com/Akanezora0/TACO-Benchmark}
}
```

## License

[MIT License](LICENSE) -->

---

<div align="center">

⭐️ **Enjoying TACO?** Give us a [star](https://github.com/Akanezora0/TACO-Benchmark/stargazers) if you'd like to support the project!

</div>
