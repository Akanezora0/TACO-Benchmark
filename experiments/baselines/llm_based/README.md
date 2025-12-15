# LLM-Based Methods Experiments

This directory contains experiments for LLM-based Text-to-SQL methods that use advanced prompting strategies and few-shot learning.

## Supported Methods

### 1. DIN-SQL
- **Description**: Decompose-and-Interpret approach for complex SQL generation
- **Key Features**: Query decomposition, step-by-step reasoning
- **Implementation**: `din_sql/`

### 2. MAC-SQL
- **Description**: Multi-Agent Collaboration for SQL generation
- **Key Features**: Multi-agent framework, collaborative generation
- **Implementation**: `mac_sql/`

## Experimental Settings

All LLM-based methods are evaluated under the same settings as baseline models:

- **Origin**: Original query + Full schema
- **QR**: Rewritten query + Full schema (with TACO-SQL Question Rewriting)
- **QR+TL**: Rewritten query + Filtered schema (with TACO-SQL components)
- **QR+TL+QP**: Full TACO-SQL framework

## Fair Comparison

To ensure fair comparison:
- **Same prompt template** as baseline models (for Origin setting)
- **Same TACO-SQL components** when applicable
- **Same evaluation metric**: Execution Accuracy (EX)
- **Same test set**: All methods evaluated on identical queries

## Running Experiments

```bash
# Run DIN-SQL experiment
python experiments/baselines/llm_based/din_sql/run_din_sql.py \
    --setting origin \
    --dataset taco_beijing \
    --output results/din_sql_origin.json

# Run MAC-SQL experiment
python experiments/baselines/llm_based/mac_sql/run_mac_sql.py \
    --setting origin \
    --dataset taco_beijing \
    --output results/mac_sql_origin.json
```

## Results

Results are saved in JSON format with the same structure as baseline experiments for easy comparison.

