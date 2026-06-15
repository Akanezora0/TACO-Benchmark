# Installation Guide

## Requirements

| Component | Version | Notes |
|:--|:--|:--|
| Python | **3.10+** | Benchmark generation, experiments, CLI |
| LLM API | OpenAI-compatible | Required for regeneration & baseline LLM runs |
| Disk space | **~10 GB+** | Dataset archive + extracted SQLite databases |

Optional:

| Component | When needed |
|:--|:--|
| CUDA GPU | SFT baselines (`requirements-sft.txt`) |
| scipy, matplotlib | Evaluation plots (`requirements-eval.txt`) |

---

## Quick install

From the repository root:

```bash
git clone https://github.com/Akanezora0/TACO-Benchmark.git
cd TACO-Benchmark

# Cross-platform (recommended)
python scripts/setup_env.py

# Linux / macOS shortcut
bash setup_env.sh
```

This creates `.venv`, installs dependencies, installs the `taco` CLI (`pip install -e .`), and copies config templates.

Activate the virtual environment:

```bash
source .venv/bin/activate   # Linux / macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell
```

Verify:

```bash
taco --version
taco info
```

---

## Configure LLM API

Copy and edit (created automatically by `setup_env.py` if missing):

```text
configs/llm_config.yaml                        # experiments & shared settings
benchmark/generation/sql_filling/config.yaml   # data-generation pipeline
configs/taco_sql_config.yaml                   # TACO-SQL ablation (optional)
```

Example (`configs/llm_config.yaml`):

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.1
  max_tokens: 8000
  api_url: "https://api.openai.com/v1"
  api_key: "your-api-key-here"
```

Environment variable overrides:

```bash
export TACO_API_KEY="sk-..."
export TACO_API_URL="https://api.openai.com/v1"
export TACO_MODEL="gpt-4o"
```

**Do not commit** real API keys. Local config files are listed in `.gitignore`.

---

## Download the dataset

The benchmark data is **not** included in git. Download from Google Drive:

| Link | File |
|:--|:--|
| [Google Drive](https://drive.google.com/file/d/1bPSYa8173XcFb1jqGQR5luNzCTxYmy_L/view?usp=sharing) | `TACO-Benchmark.zip` |

```bash
taco data download
taco data verify
```

Equivalent script:

```bash
python scripts/download_dataset.py
python scripts/download_dataset.py --verify
```

See [DATASET.md](DATASET.md) for layout and formats.

---

## Dependency profiles

```bash
# Core (default)
pip install -r requirements.txt
pip install -e .

# + evaluation plots
pip install -r requirements-eval.txt

# + SFT model baselines
pip install -r requirements-sft.txt

# Or via pyproject extras
pip install -e ".[eval]"
pip install -e ".[full]"
```

---

## Run experiments via CLI

```bash
# Baseline on official test split
taco eval run --model gpt-4o --dataset beijing

# Batch baselines
taco eval batch --models gpt-4o,deepseek-r1 --dataset beijing

# Summarize saved results
taco eval report --pred experiments/results/baseline_gpt_4o_taco_beijing.json

# TACO-SQL ablation
taco exp ablation --setting qr_tl_qp --model gpt-4o --dataset beijing

# Run all base-LLM baselines
taco exp run-all --dataset beijing --base-llm
```

Equivalent scripts remain under `experiments/` for direct invocation.

## Run a quick smoke test

After dataset download and API configuration:

```bash
# Check dataset
taco data verify

# Baseline evaluation (requires API key in config)
taco eval run --model gpt-4o --dataset beijing
```

Full experiment workflows: [docs/EXPERIMENTS.md](EXPERIMENTS.md) · Module scripts: [experiments/README.md](../experiments/README.md).

Documentation index: [docs/README.md](README.md).

Quick reproduction: `bash examples/quick_eval.sh`

---

## Troubleshooting

| Issue | Suggestion |
|:--|:--|
| `taco: command not found` | Activate `.venv` or run `pip install -e .` |
| Google Drive download fails | Install `gdown` manually: `pip install gdown`; retry `taco data download --force` |
| `Dataset incomplete` after extract | Check archive structure; see manual steps in [DATASET.md](DATASET.md) |
| Missing `benchmark/data/` | Expected — run `taco data download` first |
