# Baseline Experiments

This directory contains baseline experiments for various Text-to-SQL models, ensuring fair comparison across different model types.

## Directory Structure

```
baselines/
├── base_llm/              # Base LLM experiments
│   ├── evaluate_baseline.py    # Baseline evaluation script
│   └── README.md
├── llm_based/             # LLM-based method experiments
├── sft_based/             # SFT-based method experiments
└── hybrid/                # Hybrid method experiments
```

## Fair Comparison Principles

To ensure fair comparison across different models, we follow these principles:

### 1. Consistent Prompt Strategy
- **Same prompt template** for all models in the same experimental setting
- **Same schema format** (full schema for baseline, filtered schema for TACO-SQL)
- **Same output format requirements** (SQL only, no explanations)

### 2. Consistent Evaluation Metrics
- **Execution Accuracy (EX)**: Primary metric
- **Execution-based evaluation**: Compare execution results, not SQL strings
- **Same database environment**: All models evaluated on the same SQLite databases

### 3. Consistent Experimental Settings
- **Origin setting**: All models use original query + full schema
- **TACO-SQL settings**: All models use the same TACO-SQL components (QR, TL, QP)
- **Same temperature settings**: temperature=0.1 for SQL generation (for reproducibility)

### 4. Model-Specific Adaptations
While maintaining fairness, we adapt to model characteristics:
- **Base LLMs**: API-based, use chat completion format
- **SFT Models**: Local inference, may use schema filtering
- **Different context windows**: Handle schema truncation appropriately

## Baseline Prompt Strategy

### Prompt Template (Origin Setting)

```
You are a SQL expert. Generate SQL queries based on natural language queries and database schema.

{schema_text}

Natural language query: {query}

Requirements:
1. Generate complete, executable SQL statements
2. All table and column names must be wrapped in double quotes (including Chinese and special characters)
3. Ensure SQL syntax is correct and can be executed on SQLite
4. Output only SQL statements, no explanations or comments

Database: {database}

SQL query:
```

### Key Settings
- **Schema**: Full schema (all tables, all columns)
- **Query**: Original natural language query
- **Temperature**: 0.1 (for reproducibility)
- **Max tokens**: 2000

## Model Configurations

### Base LLMs
- **GPT-4o**: temperature=0.1, max_tokens=2000, context_window=128000
- **GPT-4o-mini**: temperature=0.1, max_tokens=2000, context_window=128000
- **GPT-o1**: temperature=0.1, max_tokens=2000, context_window=200000
- **DeepSeek-R1**: temperature=0.1, max_tokens=2000, context_window=64000

### SFT-Based Models
- **CodeS-33B**: temperature=0.1, num_beams=4, schema_filter enabled
- **Qwen2.5-Coder-32B**: temperature=0.1, num_beams=4, schema_filter enabled

## Running Baseline Experiments

```bash
# Run baseline for a specific model
python experiments/baselines/base_llm/evaluate_baseline.py \
    --model gpt-4o \
    --dataset taco_beijing \
    --output results/baseline_gpt4o.json
```

## Evaluation

Results are evaluated using execution accuracy:
- Execute generated SQL on the database
- Compare execution results with ground truth
- Calculate accuracy: correct_count / total_count

See `../evaluation/README.md` for detailed evaluation procedures.

