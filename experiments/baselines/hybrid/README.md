# Hybrid Methods Experiments

This directory contains experiments for hybrid Text-to-SQL methods that combine multiple techniques (e.g., retrieval, generation, verification).

## Supported Methods

### 1. CHESS
- **Description**: Combines retrieval, generation, and verification
- **Key Features**: Multi-stage pipeline, error correction
- **Implementation**: `chess/`

### 2. Zero-NL2SQL
- **Description**: Zero-shot approach with schema understanding
- **Key Features**: Schema-aware generation, zero-shot learning
- **Implementation**: `zero_nl2sql/`

### 3. DIAL-SQL
- **Description**: Dialogue-based SQL generation
- **Key Features**: Interactive query refinement, multi-turn dialogue
- **Implementation**: `dial_sql/`

## Experimental Settings

Hybrid methods are evaluated under the same settings:

- **Origin**: Original query + Full schema
- **QR**: Rewritten query + Full schema (with TACO-SQL Question Rewriting)
- **QR+TL**: Rewritten query + Filtered schema (with TACO-SQL components)
- **QR+TL+QP**: Full TACO-SQL framework

## Method-Specific Configurations

Each hybrid method may have unique configurations:
- **CHESS**: Retrieval top-k, verification threshold
- **Zero-NL2SQL**: Schema encoding strategy
- **DIAL-SQL**: Max dialogue turns, clarification strategy

## Fair Comparison

To ensure fair comparison:
- **Same evaluation metric**: Execution Accuracy (EX)
- **Same test set**: All methods evaluated on identical queries
- **Method-specific features documented**: Clearly indicate unique features
- **Same baseline prompt**: When applicable, use same prompt as baseline models

## Running Experiments

```bash
# Run CHESS experiment
python experiments/baselines/hybrid/chess/run_chess.py \
    --setting origin \
    --dataset taco_beijing \
    --output results/chess_origin.json

# Run Zero-NL2SQL experiment
python experiments/baselines/hybrid/zero_nl2sql/run_zero_nl2sql.py \
    --setting origin \
    --dataset taco_beijing \
    --output results/zero_nl2sql_origin.json

# Run DIAL-SQL experiment
python experiments/baselines/hybrid/dial_sql/run_dial_sql.py \
    --setting origin \
    --dataset taco_beijing \
    --output results/dial_sql_origin.json
```

## Results

Results are saved in JSON format with method-specific metadata.

