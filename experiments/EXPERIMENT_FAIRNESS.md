# Experiment Fairness and Comparison Guidelines

This document outlines the principles and practices we follow to ensure fair comparison across different Text-to-SQL models in our experiments.

## Core Principles

### 1. Consistent Experimental Settings

All models are evaluated under **identical experimental conditions**:

- **Same dataset**: All models tested on the same test set
- **Same queries**: Each model receives the exact same natural language queries
- **Same schema**: Models receive the same schema information (full schema for baseline, filtered schema for TACO-SQL)
- **Same evaluation metric**: Execution Accuracy (EX) for all models

### 2. Standardized Prompt Strategy

#### Baseline Setting (Origin)
- **Same prompt template** for all models
- **Same schema format**: Full schema with all tables and columns
- **Same output requirements**: SQL only, no explanations

#### TACO-SQL Settings
- **Same TACO-SQL components**: All models use the same Question Rewriting, Table Linking, and Query Planning
- **Same filtered schema**: All models receive the same Top-K tables from Table Linking
- **Same prompt template**: TACO-SQL prompt template applied consistently

### 3. Consistent Model Parameters

For reproducibility and fairness:

- **Temperature**: 0.1 for SQL generation (all models)
- **Max tokens**: 2000 for SQL generation (all models)
- **Same decoding strategy**: Greedy decoding for base LLMs, beam search (num_beams=4) for SFT models

### 4. Model-Specific Adaptations (While Maintaining Fairness)

While we standardize as much as possible, some adaptations are necessary due to model characteristics:

#### Base LLMs (GPT-4o, GPT-o1, etc.)
- **Format**: Chat completion API format
- **Schema handling**: Full schema (within context window limits)
- **No schema filtering**: Use complete schema information

#### SFT-Based Models (CodeS, Qwen2.5-Coder, etc.)
- **Format**: Direct text generation
- **Schema filtering**: May use schema item classifier to reduce schema size
- **Beam search**: num_beams=4 for better quality

**Note**: Schema filtering for SFT models is a technical necessity (context limits), not an unfair advantage. The filtering is applied consistently across all SFT models.

### 5. Evaluation Consistency

#### Execution-Based Evaluation
- **Same execution environment**: All SQL executed on the same SQLite database
- **Same result comparison**: Exact match comparison (row-by-row, value-by-value)
- **Same error handling**: Syntax errors and execution errors treated consistently

#### No Manual Intervention
- **Automated evaluation**: No human judgment in correctness assessment
- **Reproducible**: Same evaluation code for all models
- **Transparent**: All evaluation procedures documented and open-sourced

## Experimental Settings Comparison

| Setting | Query | Schema | Components | Models |
|---------|-------|--------|------------|--------|
| **Origin** | Original | Full | None | All |
| **QR** | Rewritten | Full | QR | All |
| **QR+TL** | Rewritten | Filtered (Top-K) | QR + TL | All |
| **QR+TL+QP** | Rewritten | Filtered (Top-K) | QR + TL + QP | All |

**Key Point**: Within each setting, all models receive identical inputs and use the same components.

## Fair Comparison Checklist

When comparing models, ensure:

- [ ] Same experimental setting (Origin, QR, QR+TL, or QR+TL+QP)
- [ ] Same test queries
- [ ] Same schema information (full or filtered consistently)
- [ ] Same evaluation metric (Execution Accuracy)
- [ ] Same evaluation procedure
- [ ] Same temperature and decoding parameters (where applicable)
- [ ] Results reported with confidence intervals or statistical significance

## Reporting Results

When reporting experimental results:

1. **Specify experimental setting**: Clearly indicate which setting (Origin, QR, QR+TL, QR+TL+QP)
2. **Report all models**: Include all models tested, not just best performers
3. **Statistical significance**: Report confidence intervals or p-values for comparisons
4. **Error analysis**: Provide breakdown of error types
5. **Reproducibility**: Include all hyperparameters and configurations

## Example: Fair Comparison Report

```
Model Comparison (Origin Setting, TACO-Beijing Test Set)

Model              | Execution Accuracy | 95% CI
-------------------|-------------------|----------
GPT-4o             | 0.35              | [0.32, 0.38]
GPT-4o-mini        | 0.28              | [0.25, 0.31]
GPT-o1             | 0.42              | [0.39, 0.45]
CodeS-33B          | 0.31              | [0.28, 0.34]
Qwen2.5-Coder-32B  | 0.29              | [0.26, 0.32]

All models evaluated on the same 1000 test queries.
Temperature: 0.1, Max tokens: 2000.
Evaluation metric: Execution Accuracy (EX).
```

## Code Organization for Fairness

Our code structure ensures fairness:

```
experiments/
├── baselines/              # Baseline experiments (same prompt for all)
│   └── base_llm/
│       └── evaluate_baseline.py    # Standardized evaluation
├── taco_sql_exp/           # TACO-SQL experiments (same components for all)
│   └── experiment_runner.py        # Unified experiment runner
└── evaluation/             # Consistent evaluation tools
    └── exec_eval.py                # Same evaluation for all models
```

## Conclusion

Fair comparison is essential for meaningful evaluation. We ensure fairness through:

1. **Standardization**: Same prompts, schemas, and evaluation procedures
2. **Transparency**: All configurations and procedures documented
3. **Reproducibility**: All code and settings open-sourced
4. **Consistency**: Same experimental conditions for all models

Any model-specific adaptations are clearly documented and justified, and do not provide unfair advantages.

