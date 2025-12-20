# TACO-SQL Core Settings and Prompt Strategies

This document provides detailed explanations of the TACO-SQL framework's experimental settings, prompt strategies, and core implementation approaches for open-source publication.

## Part I: Experimental Settings

### 1.1 Ablation Study Design

The TACO-SQL framework uses a progressive component addition ablation study design to evaluate the contribution of each component:

| Experimental Setting | Component Configuration | Description |
|---------------------|------------------------|-------------|
| **Origin** | Original query + Full schema | Baseline setting, no TACO-SQL components |
| **QR** | + Question Rewriting | Question rewriting component only |
| **QR+TL** | + Question Rewriting + Table Linking | Question rewriting and table retrieval |
| **QR+TL+QP** | + Question Rewriting + Table Linking + Query Planning | Full TACO-SQL framework |

### 1.2 Model Types Evaluated

The experiments cover four types of Text-to-SQL models:

#### Base LLMs
- GPT-4, GPT-4o, GPT-4o-mini, GPT-o1
- DeepSeek-v2.5, DeepSeek-R1
- Llama3-70b, Qwen2-72b

#### LLM-Based Methods
- DIN-SQL
- MAC-SQL

#### SFT-Based Methods
- CodeS-33B, CodeS-15B
- Qwen2.5-Coder-32B
- Deepseek-coder-6.7b

#### Hybrid Methods
- CHESS
- Zero-NL2SQL
- DIAL-SQL

### 1.3 Evaluation Metrics

**Primary Metric: Execution Accuracy (EX)**
- Definition: Proportion of queries where predicted SQL execution results exactly match ground truth results
- Formula: `EX = (Number of correct executions) / (Total number of queries)`

**Auxiliary Metrics:**
- Recall@K: Table retrieval accuracy (K=1,3,5,10)
- SQL syntax correctness rate
- Execution time

## Part II: Prompt Strategies

### 2.1 Question Rewriting

#### Functional Goals
- Clarify ambiguous or redundant user queries
- Normalize query expressions, remove redundant information
- Clarify query intent

#### Prompt Template

**System Prompt**:
```
You rewrite user questions for SQL retrieval. Remove irrelevant chatter, disambiguate entities, 
and output one concise sentence expressing the core intent while preserving key filters.
```

**Few-Shot Examples**:
```python
FEW_SHOTS = [
    (
        "I need my employee records to finish a report. Please tell me where I can get my employee records. My employee ID is E12345, Thanks.",
        "Find storage locations for employee records with ID E12345.",
    ),
    (
        "嗨，我想知道 2023 年 5 月的销售数据，最好能按地区分组，谢谢！",
        "Retrieve 2023-05 sales figures grouped by region.",
    ),
    (
        "Our customer table keeps crashing! What are the emails of users registered after 2024-01-01?",
        "List emails of customers registered after 2024-01-01.",
    ),
]
```

**Prompt Construction**:
- System message + Few-shot examples (user-assistant pairs) + Current query

#### Implementation
- **Method**: LLM-based few-shot prompting
- **Model**: GPT-4o / DeepSeek and other general-purpose large language models
- **Parameters**:
  - Temperature: 0.3 (lower temperature ensures rewriting stability)
  - Top-p: 0.9
  - Max Tokens: 512

#### Example

**Input (Original Query)**:
```
我想看看最近几年北京地区企业注册的情况，包括注册数量和注册资本
```

**Output (Rewritten)**:
```
查询北京地区企业注册数据：注册数量、注册资本，按年份统计
```

### 2.2 Table Linking

#### Functional Goals
- Identify relevant tables from large-scale heterogeneous databases
- Reduce schema size, improve SQL generation efficiency

#### Implementation Architecture

**Offline Training Phase:**
- **Model**: SBERT dual-tower model (Sentence-BERT)
- **Base Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Training Method**: Contrastive learning (Multiple Negatives Ranking Loss)
- **Training Data**: query-table pairs (positive samples)
- **Training Parameters**:
  - Epochs: 3
  - Batch Size: 16
  - Learning Rate: 2e-5
  - Warmup Steps: 100

**Online Retrieval Phase:**
- **Method**: Semantic retrieval (cosine similarity)
- **Retrieval Strategy**: Top-K retrieval (default K=5)
- **Table Information Construction**: `table description + column name list`

#### Table Information Extraction

```python
def extract_table_info(row):
    description = row['table_description']
    column_names = [col for col in row.index if 'column_content' in col and pd.notnull(row[col])]
    table_info = description + ' ' + ' '.join(column_names)
    return table_info
```

**Table Information Format**: `table description + column name list` (space-separated)

#### Training Data Construction

```python
# Build query-table pairs from training set
for query, table_names in train_data.items():
    for table_name in table_names:
        table_info = extract_table_info(merged_table[table_name])
        # Build positive sample pairs
        train_examples.append(InputExample(
            texts=[query, table_info], 
            label=1.0
        ))
```

#### Retrieval Process

```python
# 1. Encode queries and all tables
query_embeddings = query_model.encode(queries, convert_to_tensor=True)
table_embeddings = table_model.encode(table_infos, convert_to_tensor=True)

# 2. Semantic retrieval (cosine similarity)
hits = util.semantic_search(query_embedding, table_embeddings, top_k=k)

# 3. Return Top-K table names
retrieved_tables = [merged_table.iloc[hit['corpus_id']]['table_name'] for hit in hits]
```

#### Evaluation Metrics
- Recall@1, Recall@3, Recall@5, Recall@10
- Calculation: Whether retrieved tables contain ground truth relevant tables

### 2.3 Query Planning

#### Functional Goals
- Decompose complex queries into multiple simple subqueries
- Determine execution order and dependencies
- Handle cross-database queries

#### Prompt Template

```python
"""Decompose the following query into multiple simple subqueries and determine execution order.

Original Query: {query}

Relevant Tables: {relevant_tables}

Schema Information: {schema_info}

Please output the execution plan in JSON format as follows:
[
    {
        "subquery": "subquery description",
        "tables": ["table1", "table2"],
        "order": 1,
        "dependencies": []
    },
    ...
]

Execution Plan:"""
```

#### Implementation
- **Method**: LLM-based structured planning
- **Model**: GPT-4o
- **Parameters**:
  - Temperature: 0.3 (lower temperature ensures planning stability)
  - Max Tokens: 1024
- **Output Format**: JSON structured plan

#### Execution Plan Structure

```json
[
    {
        "subquery": "Query enterprise registration basic information",
        "tables": ["enterprise_registration_table"],
        "order": 1,
        "dependencies": []
    },
    {
        "subquery": "Count registration numbers and registered capital by year",
        "tables": ["enterprise_registration_table"],
        "order": 2,
        "dependencies": [1]
    }
]
```

### 2.4 SQL Generation

#### 2.4.1 Baseline Prompt (Origin Setting)

**Applicable Scenario**: Origin experimental setting (original query + full schema)

**Prompt Template**:

```python
"""You are a SQL expert. Generate SQL queries based on natural language queries and database schema.

{schema_text}

Natural Language Query: {query}

Requirements:
1. Generate complete, executable SQL statements
2. All table and column names must be wrapped in double quotes (including Chinese and special characters)
3. Ensure SQL syntax is correct and can be executed on SQLite
4. Output only SQL statements, no explanations or comments

Database: {database}

SQL Query:"""
```

**Key Settings**:
- Schema inclusion: All tables (determined by model context window)
- Table and column names: Wrapped in double quotes (supports Chinese)
- Output format: SQL statements only, no additional explanations

#### 2.4.2 TACO-SQL Prompt (After Table Linking)

**Applicable Scenario**: QR+TL and QR+TL+QP experimental settings

**Prompt Template**:

```python
"""You are a SQL expert. Generate SQL queries based on natural language queries and relevant database schema.

Relevant Table Schema Information:
{filtered_schema_text}

Natural Language Query: {rewritten_query}

Requirements:
1. Generate complete, executable SQL statements
2. All table and column names must be wrapped in double quotes (including Chinese and special characters)
3. Ensure SQL syntax is correct and can be executed on SQLite
4. Output only SQL statements, no explanations or comments
5. Use only the relevant tables listed above, do not use unlisted tables

Database: {database}

SQL Query:"""
```

**Key Improvements**:
- Schema filtering: Only includes relevant tables retrieved by Table Linking (Top-K)
- Query input: Uses rewritten query from Question Rewriting
- Table restriction: Explicitly requires using only relevant tables

#### 2.4.3 Model-Specific Configurations

**Base LLM Configuration** (GPT-4o, etc.):
```python
{
    "temperature": 0.1,  # Low temperature ensures SQL accuracy
    "max_tokens": 2000,
    "context_window": 128000  # GPT-4o large context window
}
```

**SFT Model Configuration** (CodeS, etc.):
```python
{
    "max_new_tokens": 512,
    "num_beams": 4,  # Beam search improves generation quality
    "num_return_sequences": 4  # Generate multiple candidates
}
```

**Schema Filtering Configuration** (for SFT models):
```python
{
    "enabled": true,
    "model_path": "sic_ckpts/sic_spider",  # Schema Item Classifier
    "max_tables": 7,
    "max_columns": 20
}
```

**Schema Filtering Process** (for SFT models):
1. Use Schema Item Classifier (SIC) to score table/column relevance
2. Select Top-K tables (default 7) and Top-M columns per table (default 20)
3. Build filtered schema sequence
4. Input format: `schema_sequence + content_sequence + query`

**SFT Model Input Format**:
```python
prefix_seq = (
    data["schema_sequence"] + "\n" + 
    data["content_sequence"] + "\n" + 
    data["text"] + "\n"
)
```

## Part III: Code Implementation Strategy

### 3.1 Pipeline Architecture

```python
class TACOSQLPipeline:
    def run(self, user_query: str, enable_components: List[str]) -> Dict:
        result = {
            'original_query': user_query,
            'rewritten_query': None,
            'relevant_tables': None,
            'execution_plan': None,
            'sql_queries': []
        }
        
        # Step 1: Question Rewriting
        if 'qr' in enable_components:
            result['rewritten_query'] = self.question_rewriter.rewrite(user_query)
        else:
            result['rewritten_query'] = user_query
        
        # Step 2: Table Linking
        if 'tl' in enable_components:
            result['relevant_tables'] = self.table_retriever.retrieve(
                result['rewritten_query'], 
                top_k=5
            )
        else:
            result['relevant_tables'] = []  # Use full schema
        
        # Step 3: Query Planning
        if 'qp' in enable_components and result['relevant_tables']:
            result['execution_plan'] = self.query_planner.plan(
                result['rewritten_query'],
                result['relevant_tables']
            )
        else:
            # Default single-step plan
            result['execution_plan'] = [{
                'subquery': result['rewritten_query'],
                'tables': result['relevant_tables'],
                'order': 1
            }]
        
        # Step 4: SQL Generation
        for plan_item in result['execution_plan']:
            sql = self.sql_generator.generate(
                plan_item['subquery'],
                plan_item['tables']
            )
            result['sql_queries'].append(sql)
        
        return result
```

### 3.2 Experimental Setting Mapping

| Experimental Setting | enable_components | Schema Strategy |
|---------------------|------------------|----------------|
| Origin | `[]` | Full schema |
| QR | `['qr']` | Full schema |
| QR+TL | `['qr', 'tl']` | Filtered schema (Top-K tables) |
| QR+TL+QP | `['qr', 'tl', 'qp']` | Filtered schema + Query planning |

### 3.3 Schema Formatting Strategy

#### Full Schema (Origin, QR Settings)

```python
def format_schema_simple(schema: Dict, max_tables: int = None) -> str:
    """Format full schema"""
    text = "Database Schema Information:\n\n"
    
    for table in schema.get('tables', []):
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
        text += f"Table: {table_name}\n"
        text += "  Columns:\n"
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('data_type', 'TEXT')
            text += f"    - {col_name} ({col_type})\n"
        text += "\n"
    
    return text
```

#### Filtered Schema (QR+TL, QR+TL+QP Settings)

```python
def format_schema_filtered(schema: Dict, relevant_tables: List[str]) -> str:
    """Format filtered schema (only includes relevant tables)"""
    text = "Relevant Table Schema Information:\n\n"
    
    for table in schema.get('tables', []):
        table_name = table.get('table_name', '')
        if table_name not in relevant_tables:
            continue
        
        # ... format table information ...
    
    return text
```

## Part IV: Core Settings Summary

### 4.1 Component Configuration Parameters

| Component | Key Parameter | Default | Description |
|-----------|--------------|---------|-------------|
| Question Rewriting | temperature | 0.3 | Balance creativity and accuracy |
| Question Rewriting | max_tokens | 512 | Rewritten query length limit |
| Table Linking | top_k | 5 | Number of relevant tables to retrieve |
| Table Linking | training_epochs | 3 | SBERT training epochs |
| Query Planning | temperature | 0.3 | Low temperature ensures planning stability |
| Query Planning | max_tokens | 1024 | Planning output length |
| SQL Generation | temperature | 0.1 | Low temperature ensures SQL accuracy |
| SQL Generation | max_tokens | 2000 | SQL generation length limit |

### 4.2 Experimental Workflow

```bash
# 1. Baseline experiment (Origin setting)
python experiments/baselines/base_llm/evaluate_baseline.py \
    --model gpt-4o \
    --setting origin \
    --dataset taco_beijing

# 2. TACO-SQL ablation experiment
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr_tl_qp \
    --model gpt-4o \
    --dataset taco_beijing

# 3. Evaluate results
python experiments/evaluation/exec_eval.py \
    --pred results/raw/predictions.json \
    --gold data/test.json
```

### 4.3 Key Design Decisions

1. **Schema Filtering Strategy**
   - **Problem**: Full schema contains thousands of tables, exceeding model context window
   - **Solution**: Use Table Linking to retrieve Top-K tables (default K=5)
   - **Effect**: Reduce context length by 90%+, improve model attention focus

2. **Query Rewriting Necessity**
   - **Problem**: Real-world user queries contain redundant, ambiguous expressions
   - **Solution**: Use Few-shot LLM for query rewriting
   - **Effect**: Normalize expressions, improve Table Linking and SQL generation accuracy

3. **Query Planning Applicable Scenarios**
   - **Problem**: Cross-database complex queries require multi-step execution
   - **Solution**: LLM generates structured execution plans
   - **Effect**: Decompose complex queries into simple subqueries, execute step by step

4. **Prompt Design Principles**
   - **Clear output format**: SQL statements only, no additional explanations (avoid parsing errors)
   - **Table/column name handling**: Wrapped in double quotes (supports Chinese and special characters)
   - **Clear schema structure**: Table-column hierarchy for easy model understanding
   - **Few-shot examples**: Provide high-quality examples to guide model behavior

5. **Temperature Parameter Selection**
   - **Question Rewriting**: 0.3 (balance creativity and accuracy)
   - **Query Planning**: 0.3 (ensure planning stability)
   - **SQL Generation**: 0.1 (ensure SQL accuracy, avoid syntax errors)

## Part V: File Structure

### 5.1 Core Code Locations

```
experiments/
├── baselines/                    # Baseline experiments
│   └── base_llm/
│       └── evaluate_baseline.py  # Baseline evaluation code (includes Origin setting)
├── taco_sql_exp/                 # TACO-SQL ablation experiments
│   ├── qr/                       # + Question Rewriting
│   ├── qr_tl/                    # + QR + Table Linking
│   └── qr_tl_qp/                 # Full TACO-SQL
├── evaluation/                   # Evaluation tools
│   ├── exec_eval.py              # Execution accuracy evaluation
│   └── evaluation.py             # Comprehensive evaluation
└── results/                      # Experimental results
```

### 5.2 TACO-SQL Component Code Locations

```
taco_sql/
├── question_rewriting/
│   └── rewriting.py               # Question Rewriting implementation
├── table_linking/
│   ├── training/
│   │   └── finetune_sbert.py     # SBERT training code
│   └── retrieval/
│       └── table_retrieval.py    # Table retrieval implementation
├── query_planning/
│   └── planning.py                # Query planning implementation
├── sql_generation/
│   └── text2sql.py                # SQL generation implementation
└── pipeline/
    └── taco_sql_pipeline.py      # Complete Pipeline
```

## Part VI: Experiment Reproduction Instructions

### 6.1 Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys (Base LLM)
export OPENAI_API_KEY="your_api_key"
export DEEPSEEK_API_KEY="your_api_key"
```

### 6.2 Data Preparation

```bash
# 1. Prepare test dataset
# Location: benchmark/data/final/test.json

# 2. Prepare schema files
# Location: benchmark/data/schemas/

# 3. Prepare Table Linking model (if trained)
# Location: taco_sql/table_linking/models/
# Or run training script:
python taco_sql/table_linking/training/finetune_sbert.py
```

### 6.3 Running Experiments

#### Method 1: Using Pipeline (Recommended)

```python
# Example: Run full TACO-SQL experiment
from taco_sql.pipeline.taco_sql_pipeline import TACOSQLPipeline
import yaml

# Load configuration
with open('configs/taco_sql_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create Pipeline
pipeline = TACOSQLPipeline(config=config)

# Run experiment
result = pipeline.run(
    user_query="Query enterprise registration status in Beijing area",
    enable_components=['qr', 'tl', 'qp', 'sg']
)

print(f"Generated SQL: {result['sql_queries']}")
```

#### Method 2: Using Experiment Scripts

```bash
# Baseline experiment (Origin setting)
python experiments/baselines/base_llm/evaluate_baseline.py \
    --model gpt-4o \
    --dataset taco_beijing \
    --output results/baseline_gpt4o.json

# TACO-SQL ablation experiment
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr_tl_qp \
    --model gpt-4o \
    --dataset taco_beijing \
    --output results/taco_sql_gpt4o.json
```

### 6.4 Evaluating Results

```bash
# Execution accuracy evaluation
python experiments/evaluation/exec_eval.py \
    --pred results/predictions.json \
    --gold benchmark/data/final/test.json \
    --output results/evaluation_report.json

# Result comparison and analysis
python experiments/evaluation/compare.py \
    --results_dir results/ \
    --output results/comparison_report.md
```

---

**Document Version**: v1.0  
**Last Updated**: January 2025  
**Maintainer**: TACO Project Team


