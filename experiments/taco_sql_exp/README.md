# TACO-SQL Ablation Experiments

This directory contains the implementation of TACO-SQL framework ablation experiments, including experimental settings, prompt strategies, and execution scripts.

## Directory Structure

```
taco_sql_exp/
├── prompts/                    # Prompt strategy implementations
│   ├── question_rewriting_prompt.py    # Question Rewriting Prompt
│   ├── query_planning_prompt.py        # Query Planning Prompt
│   └── sql_generation_prompt.py        # SQL Generation Prompt
├── utils/                      # Utility functions
│   └── schema_utils.py         # Schema formatting utilities
├── origin/                     # Origin experimental setting
│   └── run_origin.py
├── qr/                         # QR setting (+ Question Rewriting)
│   └── run_qr.py
├── qr_tl/                      # QR+TL setting (+ QR + Table Linking)
│   └── run_qr_tl.py
├── qr_tl_qp/                   # Full TACO-SQL (+ QR + TL + Query Planning)
│   └── run_qr_tl_qp.py
├── config.py                   # Experiment configuration
├── experiment_runner.py        # Experiment runner
├── run_ablation.py             # Main ablation experiment script
└── README.md                    # This document
```

## Experimental Settings

### 1. Origin (Baseline Setting)
- **Configuration**: Original query + Full schema
- **Components**: No TACO-SQL components
- **Purpose**: Baseline performance evaluation

### 2. QR (+ Question Rewriting)
- **Configuration**: Rewritten query + Full schema
- **Components**: Question rewriting only
- **Purpose**: Evaluate contribution of question rewriting

### 3. QR+TL (+ Question Rewriting + Table Linking)
- **Configuration**: Rewritten query + Filtered schema (Top-K tables)
- **Components**: Question rewriting + Table linking
- **Purpose**: Evaluate contribution of table linking

### 4. QR+TL+QP (Full TACO-SQL)
- **Configuration**: Rewritten query + Filtered schema + Query planning
- **Components**: Question rewriting + Table linking + Query planning
- **Purpose**: Full TACO-SQL framework performance

## Usage

### Method 1: Unified Entry Script

```bash
# Run Origin experiment
python experiments/taco_sql_exp/run_ablation.py \
    --setting origin \
    --model gpt-4o \
    --dataset taco_beijing \
    --output results/origin_gpt4o.json

# Run QR experiment
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr \
    --model gpt-4o \
    --dataset taco_beijing \
    --qr_temperature 0.3 \
    --output results/qr_gpt4o.json

# Run QR+TL experiment
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr_tl \
    --model gpt-4o \
    --dataset taco_beijing \
    --qr_temperature 0.3 \
    --tl_top_k 5 \
    --output results/qr_tl_gpt4o.json

# Run full TACO-SQL experiment
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr_tl_qp \
    --model gpt-4o \
    --dataset taco_beijing \
    --qr_temperature 0.3 \
    --tl_top_k 5 \
    --qp_temperature 0.3 \
    --output results/qr_tl_qp_gpt4o.json
```

### Method 2: Setting-Specific Scripts

```bash
# Origin
python experiments/taco_sql_exp/origin/run_origin.py \
    --model gpt-4o \
    --dataset taco_beijing

# QR
python experiments/taco_sql_exp/qr/run_qr.py \
    --model gpt-4o \
    --dataset taco_beijing \
    --qr_temperature 0.3

# QR+TL
python experiments/taco_sql_exp/qr_tl/run_qr_tl.py \
    --model gpt-4o \
    --dataset taco_beijing \
    --qr_temperature 0.3 \
    --tl_top_k 5

# QR+TL+QP
python experiments/taco_sql_exp/qr_tl_qp/run_qr_tl_qp.py \
    --model gpt-4o \
    --dataset taco_beijing \
    --qr_temperature 0.3 \
    --tl_top_k 5 \
    --qp_temperature 0.3
```

## Prompt Strategies

### Question Rewriting
- **Implementation**: `prompts/question_rewriting_prompt.py`
- **Method**: Few-shot prompting
- **Few-shot examples**: 3 examples (mixed English and Chinese)
- **Parameters**: temperature=0.3, top_p=0.9

### Query Planning
- **Implementation**: `prompts/query_planning_prompt.py`
- **Method**: Structured JSON output
- **Parameters**: temperature=0.3, max_tokens=1024

### SQL Generation
- **Implementation**: `prompts/sql_generation_prompt.py`
- **Strategies**:
  - Baseline Prompt (Origin setting)
  - TACO-SQL Prompt (QR+TL, QR+TL+QP settings)
- **Parameters**: temperature=0.1, max_tokens=2000

For detailed prompt strategies, see [TACO-SQL Core Settings](../docs/TACO-SQL_CORE_SETTINGS_AND_PROMPT_STRATEGIES.md).

## Configuration

### Model Configuration

Supported model types:
- **Base LLMs**: gpt-4o, gpt-4o-mini, gpt-o1, deepseek-r1
- **SFT-Based**: codes-33b

Model configurations are defined in `config.py` and can be viewed via the `MODEL_CONFIGS` dictionary.

### Component Parameters

| Component | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| Question Rewriting | qr_temperature | 0.3 | Rewriting temperature |
| Table Linking | tl_top_k | 5 | Top-K tables to retrieve |
| Query Planning | qp_temperature | 0.3 | Planning temperature |
| SQL Generation | sg_temperature | 0.1 | SQL generation temperature |

## Output Format

Experimental results are saved in JSON format, each result contains:

```json
{
    "item_id": "query_id",
    "original_query": "original query",
    "rewritten_query": "rewritten query (if used)",
    "relevant_tables": ["list of relevant tables"],
    "execution_plan": [{"subquery": "...", "tables": [...], "order": 1}],
    "generated_sql": "generated SQL",
    "ground_truth_sql": "ground truth SQL",
    "database": "database name",
    "schema_info": {"included_tables_count": 5, ...},
    "errors": []
}
```

## Notes

1. **Current Implementation Status**: Script framework is complete, but LLM calling parts are placeholder implementations (marked with TODO)
2. **Data Paths**: Test data and schema files need to be prepared
3. **Model Configuration**: API keys and model access information need to be configured
4. **Table Linking Model**: Pre-trained SBERT model is required (if available)

## Related Documentation

- [TACO-SQL Core Settings and Prompt Strategies](../docs/TACO-SQL_CORE_SETTINGS_AND_PROMPT_STRATEGIES.md)
- [Experiment Framework README](../README.md)

