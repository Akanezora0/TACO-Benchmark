# SFT-Based Methods Experiments

This directory contains experiments for Supervised Fine-Tuned (SFT) Text-to-SQL models that are specifically trained on SQL generation tasks.

## Supported Models

### 1. CodeS-33B / CodeS-15B
- **Description**: Code generation model fine-tuned for SQL
- **Key Features**: Schema filtering, beam search decoding
- **Implementation**: `codes/`

### 2. Qwen2.5-Coder-32B
- **Description**: Qwen code generation model fine-tuned for SQL
- **Key Features**: Large context window, instruction following
- **Implementation**: `qwen_coder/`

### 3. Deepseek-coder-6.7b
- **Description**: DeepSeek code generation model
- **Key Features**: Efficient inference, good code understanding
- **Implementation**: `deepseek_coder/`

### 4. Granite-34b-code
- **Description**: IBM Granite code generation model
- **Key Features**: Large model, strong code generation
- **Implementation**: `granite/`

## Experimental Settings

SFT models are evaluated under the same settings, with schema filtering applied when necessary:

- **Origin**: Original query + Full schema (or filtered if context limit)
- **QR**: Rewritten query + Full schema
- **QR+TL**: Rewritten query + Filtered schema (Top-K tables)
- **QR+TL+QP**: Full TACO-SQL framework

## Model-Specific Configurations

### Schema Filtering
SFT models may use Schema Item Classifier (SIC) to filter relevant tables/columns:
- **Max tables**: 7 (default)
- **Max columns per table**: 20 (default)
- **SIC model**: Pre-trained on Spider dataset

### Decoding Strategy
- **Beam search**: num_beams=4 (default)
- **Temperature**: 0.1 (for reproducibility)
- **Max tokens**: 512-2000 (model dependent)

## Fair Comparison

To ensure fair comparison with other model types:
- **Same evaluation metric**: Execution Accuracy (EX)
- **Same test set**: All models evaluated on identical queries
- **Schema filtering documented**: Clearly indicate when schema filtering is used
- **Same prompt template**: When applicable, use same prompt as baseline models

## Running Experiments

```bash
# Run CodeS-33B experiment
python experiments/baselines/sft_based/codes/run_codes.py \
    --model codes-33b \
    --dataset taco_beijing \
    --output results/codes_33b_origin.json

# Run Qwen2.5-Coder experiment
python experiments/baselines/sft_based/qwen_coder/run_qwen_coder.py \
    --model qwen2.5-coder-32b \
    --dataset taco_beijing \
    --output results/qwen_coder_origin.json
```

## Results

Results are saved in JSON format with model-specific metadata (e.g., schema filtering stats, beam search info).

