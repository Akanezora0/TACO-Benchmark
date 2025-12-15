# Evaluation Framework

This directory contains evaluation tools for Text-to-SQL experiments, ensuring consistent and fair evaluation across all models.

## Directory Structure

```
evaluation/
├── evaluation.py          # Main evaluation script
├── exec_eval.py           # Execution-based evaluation
├── exec_eval_helper.py    # Execution evaluation helper functions
├── compare.py             # Result comparison and analysis
├── draw_result.py         # Visualization tools
├── average_token.py       # Token statistics
└── tex_table.py           # LaTeX table generation
```

## Evaluation Metrics

### Primary Metric: Execution Accuracy (EX)

**Definition**: The proportion of queries where the predicted SQL execution result exactly matches the ground truth execution result.

**Formula**: 
```
EX = (Number of correct executions) / (Total number of queries)
```

**Implementation**: 
- Execute both predicted SQL and ground truth SQL on the database
- Compare execution results (row-by-row, value-by-value)
- Count exact matches

### Why Execution Accuracy?

1. **Fair Comparison**: Different SQL syntax can produce the same results
2. **Practical Relevance**: End users care about correct results, not SQL syntax
3. **Standard Practice**: Aligned with BIRD, Spider, and other Text-to-SQL benchmarks

## Evaluation Procedure

### Step 1: Execute SQL Queries

```python
from evaluation.exec_eval import execute_sql, compare_results

# Execute predicted SQL
pred_success, pred_results, pred_error = execute_sql(db_path, predicted_sql)

# Execute ground truth SQL
gt_success, gt_results, gt_error = execute_sql(db_path, ground_truth_sql)
```

### Step 2: Compare Results

```python
# Compare execution results
is_correct = compare_results(pred_results, gt_results)
```

### Step 3: Calculate Metrics

```python
# Calculate execution accuracy
total = len(test_data)
correct = sum(1 for result in results if result['is_correct'])
execution_accuracy = correct / total
```

## Fair Evaluation Principles

### 1. Consistent Execution Environment
- **Same database**: All queries executed on the same SQLite database
- **Same SQLite version**: Ensure consistent behavior
- **Same execution settings**: No special flags or configurations

### 2. Result Normalization
- **Type normalization**: Convert all values to strings for comparison
- **Null handling**: Treat NULL values consistently
- **Order independence**: Compare as sets, not ordered lists

### 3. Error Handling
- **Syntax errors**: Count as incorrect
- **Execution errors**: Count as incorrect
- **Timeout**: Count as incorrect (if applicable)

## Usage

### Basic Evaluation

```bash
# Evaluate predictions against ground truth
python experiments/evaluation/exec_eval.py \
    --pred results/predictions.json \
    --gold benchmark/data/final/test.json \
    --output results/evaluation_report.json
```

### Result Comparison

```bash
# Compare results from different experiments
python experiments/evaluation/compare.py \
    --results_dir results/ \
    --output results/comparison_report.md
```

### Visualization

```bash
# Generate visualization plots
python experiments/evaluation/draw_result.py \
    --results results/evaluation_report.json \
    --output results/visualizations/
```

## Output Format

Evaluation results are saved in JSON format:

```json
{
    "total_queries": 1000,
    "correct_executions": 350,
    "execution_accuracy": 0.35,
    "errors": {
        "syntax_errors": 50,
        "execution_errors": 100,
        "wrong_results": 500
    },
    "per_query_results": [
        {
            "item_id": "query_001",
            "predicted_sql": "SELECT ...",
            "ground_truth_sql": "SELECT ...",
            "is_correct": true,
            "error_type": null
        },
        ...
    ]
}
```

## Model Fairness

To ensure fair evaluation:

1. **Same evaluation procedure**: All models evaluated using the same code
2. **Same test set**: All models evaluated on the same test queries
3. **Same ground truth**: All models compared against the same ground truth SQL
4. **No manual intervention**: Automated evaluation, no human judgment

## Related Documentation

- [Baseline Experiments](../baselines/README.md)
- [TACO-SQL Experiments](../taco_sql_exp/README.md)

