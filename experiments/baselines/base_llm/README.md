# Baseline LLM Experiment Framework

## Directory Overview

The `experiments/baselines/base_llm/` directory contains code and results for baseline experiments, separated from the benchmark generation pipeline.

## File Organization

### Code Files
- `evaluate_baseline.py`: Baseline evaluation script (supports concurrency)

### Result Files
- `results/`: Evaluation result JSON files
- `*.log`: Evaluation log files (evaluation-related logs are stored here)

### Notes
- **NL query generation logs** should be placed in `benchmark/generation/nl_query/` directory
- **Evaluation-related logs** are placed in `experiments/baselines/base_llm/` directory

## Design Principles

1. **Simple and Direct**: No complex rule matching or keyword extraction
2. **Sufficient Context**: Include as many tables as possible based on model's context window
3. **Direct Text-to-SQL**: Let the model directly perform Text-to-SQL conversion without additional processing
4. **Concurrent Acceleration**: Use ThreadPoolExecutor to accelerate API calls

## Usage

### 1. Generate NL Queries (Run First)

```bash
cd /home/u2023103807/TACO

# Generate NL queries for a single database (logs in benchmark/generation/nl_query/)
python3 benchmark/generation/nl_query/4generate_nl_queries_improved.py \
  --sql_dir benchmark/data/beijing/output/single \
  --schema_dir benchmark/data/beijing/database_chinese \
  --output_dir benchmark/data/beijing/output/nl_query \
  --database 社会保障 \
  --max_workers 5

# Or use batch script
bash benchmark/generation/nl_query/generate_nl_for_databases.sh
```

### 2. Baseline Evaluation

```bash
cd /home/u2023103807/TACO

# Single database evaluation
python3 experiments/baselines/base_llm/evaluate_baseline.py \
  --nl_query_dir benchmark/data/beijing/output/nl_query/社会保障 \
  --sql_dir benchmark/data/beijing/output/single/社会保障 \
  --db_path benchmark/data/beijing/database_chinese/社会保障/社会保障.db \
  --schema_file benchmark/data/beijing/database_chinese/社会保障/社会保障.json \
  --model gpt-4o \
  --output_file experiments/baselines/base_llm/results/beijing_社会保障_gpt4o_baseline.json \
  --max_tables 100 \
  --max_columns_per_table 30 \
  --limit 100 \
  --max_workers 5

# Or use batch script
bash experiments/baselines/base_llm/run_baseline_eval.sh
```

### Parameter Description

- `--max_tables`: Maximum number of tables (adjust based on model context window)
  - GPT-4o (128K tokens): Recommended 100-150 tables
  - GPT-4 (8K tokens): Recommended 20-30 tables
- `--max_columns_per_table`: Maximum columns per table (recommended 20-30)
- `--limit`: Limit evaluation count (for testing)
- `--max_workers`: Number of concurrent threads (default 5, adjust based on API rate limits)

## Configuration

### Model Context Windows

- GPT-4: 8K tokens
- GPT-4o: 128K tokens
- GPT-o1: 200K tokens
- DeepSeek-R1: 64K tokens

### Schema Inclusion Strategy

- Simple and direct: Include first N tables (N adjusted based on context window)
- No complex table selection or keyword matching
- Let the model choose from sufficient context

## Output Format

Evaluation results include:
- Basic statistics (total count, execution success rate, result matching rate, etc.)
- Configuration information (table count, column count, token estimation, etc.)
- Detailed results (evaluation results for each query)

## Concurrency Mechanism

- NL query generation: Use ThreadPoolExecutor, each thread has independent OpenAI client
- Baseline evaluation: Use ThreadPoolExecutor, thread-safe client management
- Default concurrency: 5 (adjustable via `--max_workers`)

## Comparison with Paper Results

Baseline results in the paper (without TACO-SQL framework):
- GPT-4o (beijing): 12.06%

Goal: By providing sufficient context, we expect to achieve results close to the baseline results in the paper.
