# Reviewer Guide - TACO Experiments Framework

This guide helps reviewers quickly understand the experimental framework, settings, and prompt strategies.

## Quick Overview

### Experimental Framework Structure

```
experiments/
├── baselines/          # Baseline model experiments (Origin setting)
├── taco_sql_exp/      # TACO-SQL ablation experiments (4 settings)
└── evaluation/        # Evaluation tools and metrics
```

## Experimental Settings

### Baseline Experiments

**Setting**: Origin (original query + full schema)

**Models**: Base LLMs, LLM-Based, SFT-Based, Hybrid methods

**Prompt**: Standardized baseline prompt (see `baselines/base_llm/prompt_strategy.py`)

### TACO-SQL Ablation Experiments

| Setting | Query | Schema | Components | Prompt |
|---------|-------|--------|------------|--------|
| **Origin** | Original | Full | None | Baseline |
| **QR** | Rewritten | Full | Question Rewriting | Baseline |
| **QR+TL** | Rewritten | Filtered | QR + Table Linking | TACO-SQL |
| **QR+TL+QP** | Rewritten | Filtered | QR + TL + Query Planning | TACO-SQL |

## Prompt Strategies

### 1. Baseline Prompt
**Location**: `baselines/base_llm/prompt_strategy.py`
**Template**: Standard SQL generation prompt with schema and query
**Parameters**: temperature=0.1, max_tokens=2000

### 2. Question Rewriting Prompt
**Location**: `taco_sql_exp/prompts/question_rewriting_prompt.py`
**Method**: Few-shot prompting (3 examples)
**Parameters**: temperature=0.3, top_p=0.9

### 3. Query Planning Prompt
**Location**: `taco_sql_exp/prompts/query_planning_prompt.py`
**Method**: Structured JSON output
**Parameters**: temperature=0.3, max_tokens=1024

### 4. SQL Generation Prompt (TACO-SQL)
**Location**: `taco_sql_exp/prompts/sql_generation_prompt.py`
**Template**: Similar to baseline but with filtered schema
**Parameters**: temperature=0.1, max_tokens=2000

## Key Files for Review

### Core Framework
- `experiments/README.md` - Main documentation
- `experiments/FINAL_VERSION_SUMMARY.md` - Complete summary
- `experiments/docs/EXPERIMENT_FAIRNESS.md` - Fair comparison principles

### Prompt Strategies
- `baselines/base_llm/prompt_strategy.py` - Baseline prompt
- `taco_sql_exp/prompts/` - All TACO-SQL prompt strategies
- `experiments/docs/TACO-SQL_CORE_SETTINGS_AND_PROMPT_STRATEGIES.md` - Detailed prompt documentation

### Experimental Settings
- `taco_sql_exp/config.py` - Experiment configuration
- `taco_sql_exp/experiment_runner.py` - Main experiment runner
- `baselines/base_llm/experiment_config.py` - Baseline configuration

### Evaluation
- `evaluation/README.md` - Evaluation documentation
- `evaluation/metrics_calculator.py` - Metrics calculation
- `evaluation/exec_eval.py` - Execution evaluation

## Fair Comparison

- **Same Prompt**: All models in same setting use identical prompt
- **Same Parameters**: temperature=0.1, max_tokens=2000
- **Same Evaluation**: Execution Accuracy (EX)
- **Same Test Set**: All models on identical queries

See `experiments/docs/EXPERIMENT_FAIRNESS.md` for details.

## Evaluation Metric

**Primary Metric**: Execution Accuracy (EX)
- Definition: Proportion of queries where predicted SQL execution result matches ground truth
- Formula: EX = (Correct executions) / (Total queries)

