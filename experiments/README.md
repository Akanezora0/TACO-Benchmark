# TACO Experiments Framework

This directory contains the complete experimental framework for evaluating Text-to-SQL models on the TACO benchmark, including baseline experiments and TACO-SQL ablation studies.

## Directory Structure

```
experiments/
├── baselines/              # Baseline model experiments
│   ├── base_llm/          # Base LLM models (GPT-4o, GPT-o1, DeepSeek-R1, etc.)
│   ├── llm_based/         # LLM-based methods (DIN-SQL, MAC-SQL)
│   ├── sft_based/         # SFT-based models (CodeS, Qwen2.5-Coder, etc.)
│   └── hybrid/            # Hybrid methods (CHESS, Zero-NL2SQL, DIAL-SQL)
├── taco_sql_exp/          # TACO-SQL ablation experiments
│   ├── origin/            # Origin setting (baseline)
│   ├── qr/                # + Question Rewriting
│   ├── qr_tl/             # + QR + Table Linking
│   └── qr_tl_qp/          # Full TACO-SQL (+ QR + TL + Query Planning)
├── evaluation/            # Evaluation tools and metrics
└── results/               # Experimental results
```

## Experimental Settings

### Baseline Experiments

**Setting**: Origin (original query + full schema)

**Models Evaluated**:
- **Base LLMs**: GPT-4o, GPT-4o-mini, GPT-o1, DeepSeek-R1, Llama3-70b, Qwen2-72b
- **LLM-Based**: DIN-SQL, MAC-SQL
- **SFT-Based**: CodeS-33B, Qwen2.5-Coder-32B, Deepseek-coder-6.7b
- **Hybrid**: CHESS, Zero-NL2SQL, DIAL-SQL

**Prompt Strategy**: Standardized baseline prompt (see `baselines/base_llm/prompt_strategy.py`)

### TACO-SQL Ablation Experiments

Four experimental settings with progressive component addition:

1. **Origin**: Original query + Full schema
2. **QR**: + Question Rewriting
3. **QR+TL**: + Question Rewriting + Table Linking
4. **QR+TL+QP**: Full TACO-SQL (+ Question Rewriting + Table Linking + Query Planning)

**Models**: All baseline models evaluated under each setting

**Prompt Strategies**: See `taco_sql_exp/prompts/` for detailed prompt implementations

## Key Features

### Fair Comparison
- **Standardized Prompts**: Same prompt template for all models in the same setting
- **Consistent Evaluation**: Execution Accuracy (EX) as primary metric
- **Same Test Set**: All models evaluated on identical test queries
- **Documented Adaptations**: Model-specific adaptations clearly documented

### Comprehensive Evaluation
- **Execution Accuracy**: Primary metric (execution result matching)
- **Error Analysis**: Detailed error categorization and analysis
- **Statistical Rigor**: Confidence intervals and significance testing
- **Performance Metrics**: Latency, token usage, etc.

## Quick Start

### Run Baseline Experiment

```bash
# Single model
python experiments/baselines/base_llm/run_experiment.py \
    --model gpt-4o \
    --test_data benchmark/data/final/test.json \
    --output results/baseline_gpt4o.json

# Batch evaluation
python experiments/baselines/base_llm/batch_evaluate.py \
    --models gpt-4o gpt-o1 deepseek-r1 \
    --test_data benchmark/data/final/test.json
```

### Run TACO-SQL Ablation

```bash
# Full TACO-SQL experiment
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr_tl_qp \
    --model gpt-4o \
    --dataset taco_beijing \
    --output results/qr_tl_qp_gpt4o.json
```

### Evaluate Results

```bash
# Execution accuracy evaluation
python experiments/evaluation/exec_eval.py \
    --pred results/predictions.json \
    --gold benchmark/data/final/test.json \
    --output results/evaluation_report.json
```

## Documentation

- **[TACO-SQL Core Settings and Prompt Strategies](TACO-SQL_CORE_SETTINGS_AND_PROMPT_STRATEGIES.md)**: Detailed prompt strategies and experimental settings
- **[Experiment Fairness](EXPERIMENT_FAIRNESS.md)**: Fair comparison principles and practices
- **[Reviewer Guide](REVIEWER_GUIDE.md)**: Quick guide for reviewers
- **[Baseline Experiments](baselines/README.md)**: Baseline experiment documentation
- **[TACO-SQL Experiments](taco_sql_exp/README.md)**: TACO-SQL ablation experiment documentation
- **[Evaluation Framework](evaluation/README.md)**: Evaluation tools and metrics documentation

## Evaluation Metrics

### Primary Metric: Execution Accuracy (EX)

**Definition**: Proportion of queries where predicted SQL execution result exactly matches ground truth execution result.

**Formula**: `EX = (Correct executions) / (Total queries)`

### Additional Metrics

- **Error Breakdown**: Syntax errors, execution errors, wrong results
- **Performance Metrics**: Latency, token usage
- **Statistical Analysis**: Confidence intervals, significance testing

## Model Configurations

All models use standardized configurations for fair comparison:

- **Temperature**: 0.1 (for reproducibility)
- **Max Tokens**: 2000
- **Evaluation**: Execution-based (not string matching)

Model-specific configurations are documented in respective module READMEs.

## Results Structure

Results are saved in JSON format with the following structure:

```json
{
    "model": "gpt-4o",
    "setting": "qr_tl_qp",
    "dataset": "taco_beijing",
    "total_queries": 1000,
    "execution_accuracy": 0.35,
    "confidence_interval": [0.32, 0.38],
    "per_query_results": [...],
    "error_breakdown": {...}
}
```

## Notes

- All code and documentation are in English for international reviewers
- Prompt strategies are clearly documented and accessible
- Experimental settings are standardized for fair comparison
- Evaluation procedures are consistent across all models
