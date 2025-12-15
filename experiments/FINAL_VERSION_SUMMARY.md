# TACO Experiments Framework - Final Version Summary

## Overview

This document provides a final summary of the experimental framework for reviewers. The framework demonstrates comprehensive experimental work with clear experimental settings, prompt strategies, and fair comparison practices.

## Core Framework Structure

### 1. Baseline Experiments (`baselines/`)

**Purpose**: Evaluate baseline performance of various Text-to-SQL models

**Experimental Setting**: Origin (original query + full schema)

**Models Covered**:
- Base LLMs: GPT-4o, GPT-4o-mini, GPT-o1, DeepSeek-R1, Llama3-70b, Qwen2-72b
- LLM-Based: DIN-SQL, MAC-SQL
- SFT-Based: CodeS-33B, Qwen2.5-Coder-32B, Deepseek-coder-6.7b
- Hybrid: CHESS, Zero-NL2SQL, DIAL-SQL

**Key Files**:
- `baselines/base_llm/prompt_strategy.py` - Standardized prompt template
- `baselines/base_llm/model_wrapper.py` - Unified model interface
- `baselines/base_llm/run_experiment.py` - Experiment runner
- `baselines/README.md` - Detailed documentation

**Prompt Strategy**: See `baselines/base_llm/prompt_strategy.py` for standardized baseline prompt template.

---

### 2. TACO-SQL Ablation Experiments (`taco_sql_exp/`)

**Purpose**: Evaluate TACO-SQL framework components through ablation study

**Experimental Settings** (progressive component addition):
1. **Origin**: Original query + Full schema
2. **QR**: + Question Rewriting
3. **QR+TL**: + Question Rewriting + Table Linking
4. **QR+TL+QP**: Full TACO-SQL (+ Question Rewriting + Table Linking + Query Planning)

**Key Files**:
- `taco_sql_exp/prompts/question_rewriting_prompt.py` - Question Rewriting prompt strategy
- `taco_sql_exp/prompts/query_planning_prompt.py` - Query Planning prompt strategy
- `taco_sql_exp/prompts/sql_generation_prompt.py` - SQL Generation prompt strategies
- `taco_sql_exp/config.py` - Experiment configuration
- `taco_sql_exp/experiment_runner.py` - Main experiment runner
- `taco_sql_exp/README.md` - Detailed documentation

**Prompt Strategies**: 
- Question Rewriting: Few-shot prompting with 3 examples
- Query Planning: Structured JSON output format
- SQL Generation: Baseline prompt (Origin) and TACO-SQL prompt (QR+TL, QR+TL+QP)

**Detailed Documentation**: See `TACO-SQL实验核心设置与Prompt策略.md` for complete prompt strategies.

---

### 3. Evaluation Framework (`evaluation/`)

**Purpose**: Consistent and comprehensive evaluation across all experiments

**Key Files**:
- `evaluation/exec_eval.py` - Execution-based evaluation
- `evaluation/evaluation.py` - Main evaluation script
- `evaluation/evaluation_config.py` - Standardized evaluation configuration
- `evaluation/metrics_calculator.py` - Comprehensive metrics calculation
- `evaluation/error_analysis.py` - Error analysis tools
- `evaluation/README.md` - Detailed documentation

**Primary Metric**: Execution Accuracy (EX) - execution result matching

**Additional Features**:
- Statistical significance testing
- Confidence intervals
- Error categorization
- Performance metrics

---

## Experimental Settings Summary

### Baseline Setting (Origin)

**Configuration**:
- Query: Original natural language query
- Schema: Full schema (all tables, all columns)
- Components: None (baseline)

**Prompt Template**: Standardized baseline prompt (see `baselines/base_llm/prompt_strategy.py`)

**Models**: All baseline models evaluated

### TACO-SQL Settings

| Setting | Query | Schema | Components | Prompt Strategy |
|---------|-------|--------|------------|----------------|
| **Origin** | Original | Full | None | Baseline prompt |
| **QR** | Rewritten | Full | Question Rewriting | Baseline prompt |
| **QR+TL** | Rewritten | Filtered (Top-K) | QR + Table Linking | TACO-SQL prompt |
| **QR+TL+QP** | Rewritten | Filtered (Top-K) | QR + TL + Query Planning | TACO-SQL prompt |

---

## Prompt Strategies

### 1. Baseline Prompt (Origin Setting)

**Location**: `baselines/base_llm/prompt_strategy.py`

**Template**:
```
You are a SQL expert. Generate SQL queries based on natural language queries and database schema.

{schema_text}

Natural language query: {query}

Requirements:
1. Generate complete, executable SQL statements
2. All table and column names must be wrapped in double quotes
3. Ensure SQL syntax is correct and can be executed on SQLite
4. Output only SQL statements, no explanations or comments

Database: {database}

SQL query:
```

**Parameters**: temperature=0.1, max_tokens=2000

### 2. Question Rewriting Prompt

**Location**: `taco_sql_exp/prompts/question_rewriting_prompt.py`

**Method**: Few-shot prompting

**Few-shot Examples**: 3 examples (mixed English and Chinese)

**System Prompt**: "You rewrite user questions for SQL retrieval. Remove irrelevant chatter, disambiguate entities, and output one concise sentence expressing the core intent while preserving key filters."

**Parameters**: temperature=0.3, top_p=0.9

### 3. Query Planning Prompt

**Location**: `taco_sql_exp/prompts/query_planning_prompt.py`

**Method**: Structured JSON output

**Output Format**: JSON array with subquery, tables, order, dependencies

**Parameters**: temperature=0.3, max_tokens=1024

### 4. SQL Generation Prompt (TACO-SQL)

**Location**: `taco_sql_exp/prompts/sql_generation_prompt.py`

**Template**: Similar to baseline but with filtered schema and rewritten query

**Key Differences**:
- Uses rewritten query (from Question Rewriting)
- Uses filtered schema (from Table Linking, Top-K tables)
- Explicitly restricts to relevant tables only

**Parameters**: temperature=0.1, max_tokens=2000

---

## Fair Comparison Principles

### Standardization

1. **Same Prompt Template**: All models in the same setting use identical prompt template
2. **Same Parameters**: temperature=0.1, max_tokens=2000 for SQL generation
3. **Same Evaluation**: Execution Accuracy (EX) for all models
4. **Same Test Set**: All models evaluated on identical test queries

### Model-Specific Adaptations

While maintaining fairness, some adaptations are necessary:

- **Base LLMs**: API-based, use chat completion format
- **SFT Models**: Local inference, may use schema filtering (technical necessity)
- **Different Context Windows**: Schema truncation handled appropriately

All adaptations are clearly documented in `EXPERIMENT_FAIRNESS.md`.

---

## File Organization

### Core Framework Files

```
experiments/
├── README.md                           # Main documentation
├── EXPERIMENT_FAIRNESS.md              # Fair comparison principles
├── TACO-SQL实验核心设置与Prompt策略.md    # Detailed prompt strategies
│
├── baselines/                          # Baseline experiments
│   ├── README.md                       # Baseline documentation
│   ├── base_llm/
│   │   ├── prompt_strategy.py          # Standardized prompt
│   │   ├── model_wrapper.py            # Unified interface
│   │   ├── run_experiment.py           # Experiment runner
│   │   └── experiment_config.py        # Model configurations
│   ├── llm_based/                      # DIN-SQL, MAC-SQL
│   ├── sft_based/                      # CodeS, Qwen2.5-Coder
│   └── hybrid/                         # CHESS, Zero-NL2SQL
│
├── taco_sql_exp/                       # TACO-SQL ablation
│   ├── README.md                       # TACO-SQL documentation
│   ├── prompts/                        # Prompt strategies
│   │   ├── question_rewriting_prompt.py
│   │   ├── query_planning_prompt.py
│   │   └── sql_generation_prompt.py
│   ├── config.py                       # Experiment configuration
│   ├── experiment_runner.py            # Main runner
│   └── [origin|qr|qr_tl|qr_tl_qp]/     # Setting-specific scripts
│
└── evaluation/                          # Evaluation framework
    ├── README.md                       # Evaluation documentation
    ├── evaluation_config.py            # Evaluation configuration
    ├── metrics_calculator.py           # Metrics calculation
    ├── error_analysis.py               # Error analysis
    └── exec_eval.py                    # Execution evaluation
```

---

## Key Points for Reviewers

### 1. Comprehensive Experimental Coverage

- **Multiple Model Types**: Base LLMs, LLM-Based, SFT-Based, Hybrid methods
- **Multiple Settings**: Baseline + 3 TACO-SQL ablation settings
- **Fair Comparison**: Standardized prompts and evaluation procedures

### 2. Clear Prompt Strategies

- **Documented**: All prompt templates clearly documented
- **Accessible**: Prompt strategies in dedicated files
- **Consistent**: Same prompt for same experimental setting

### 3. Rigorous Evaluation

- **Primary Metric**: Execution Accuracy (EX)
- **Statistical Rigor**: Confidence intervals, significance testing
- **Error Analysis**: Detailed error categorization

### 4. Transparency

- **Documentation**: Comprehensive documentation in English
- **Code Organization**: Clear structure, easy to navigate
- **Fair Comparison**: Explicit principles and practices documented

---

## Quick Reference

### Experimental Settings

| Setting | Components | Schema | Prompt |
|---------|------------|--------|--------|
| Origin | None | Full | Baseline |
| QR | Question Rewriting | Full | Baseline |
| QR+TL | QR + Table Linking | Filtered | TACO-SQL |
| QR+TL+QP | QR + TL + Query Planning | Filtered | TACO-SQL |

### Prompt Locations

- **Baseline Prompt**: `baselines/base_llm/prompt_strategy.py`
- **Question Rewriting**: `taco_sql_exp/prompts/question_rewriting_prompt.py`
- **Query Planning**: `taco_sql_exp/prompts/query_planning_prompt.py`
- **SQL Generation**: `taco_sql_exp/prompts/sql_generation_prompt.py`

### Documentation

- **Main README**: `experiments/README.md`
- **Fair Comparison**: `experiments/EXPERIMENT_FAIRNESS.md`
- **Prompt Strategies**: `experiments/TACO-SQL实验核心设置与Prompt策略.md`
- **Baseline Docs**: `experiments/baselines/README.md`
- **TACO-SQL Docs**: `experiments/taco_sql_exp/README.md`
- **Evaluation Docs**: `experiments/evaluation/README.md`

---

## Status

✅ **Framework Complete**: All core components implemented
✅ **Documentation Complete**: Comprehensive documentation in English
✅ **Prompt Strategies Documented**: All prompt templates clearly documented
✅ **Fair Comparison Ensured**: Standardized procedures and clear documentation

**Ready for Review**: The experimental framework is complete and ready for reviewer examination.

---

**Last Updated**: 2025-01-XX
**Version**: Final

