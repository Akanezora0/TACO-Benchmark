# Baseline LLM实验框架

## 目录说明

`experiments/baselines/base_llm/` 目录用于存放Baseline实验相关的代码和结果，与benchmark生成流程分离。

## 文件组织

### 代码文件
- `evaluate_baseline.py`: Baseline评测脚本（支持并发）

### 结果文件
- `results/`: 评测结果JSON文件
- `*.log`: 评测日志文件（评测相关的日志放在这里）

### 注意
- **NL查询生成的日志**应放在 `benchmark/generation/nl_query/` 目录下
- **评测相关的日志**放在 `experiments/baselines/base_llm/` 目录下

## 设计原则

1. **简单直接**：不使用复杂的规则匹配或关键词提取
2. **给足上下文**：根据模型的上下文窗口，包含尽可能多的表信息
3. **直接Text-to-SQL**：让模型直接进行Text-to-SQL转换，不做额外处理
4. **并发加速**：使用ThreadPoolExecutor加速API调用

## 使用方法

### 1. 生成NL查询（先运行）

```bash
cd /home/u2023103807/TACO

# 为单个数据库生成NL查询（日志在benchmark/generation/nl_query/）
python3 benchmark/generation/nl_query/4generate_nl_queries_improved.py \
  --sql_dir benchmark/data/beijing/output/single \
  --schema_dir benchmark/data/beijing/database_chinese \
  --output_dir benchmark/data/beijing/output/nl_query \
  --database 社会保障 \
  --max_workers 5

# 或使用批量脚本
bash benchmark/generation/nl_query/generate_nl_for_databases.sh
```

### 2. Baseline评测

```bash
cd /home/u2023103807/TACO

# 单个数据库评测
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

# 或使用批量脚本
bash experiments/baselines/base_llm/run_baseline_eval.sh
```

### 参数说明

- `--max_tables`: 最大表数量（根据模型上下文窗口调整）
  - GPT-4o (128K tokens): 建议100-150个表
  - GPT-4 (8K tokens): 建议20-30个表
- `--max_columns_per_table`: 每个表最大列数（建议20-30）
- `--limit`: 限制评测数量（用于测试）
- `--max_workers`: 并发线程数（默认5，可根据API限流调整）

## 配置说明

### 模型上下文窗口

- GPT-4: 8K tokens
- GPT-4o: 128K tokens
- GPT-o1: 200K tokens
- DeepSeek-R1: 64K tokens

### Schema包含策略

- 简单直接：包含前N个表（N根据上下文窗口调整）
- 不进行复杂的表选择或关键词匹配
- 让模型从足够的上下文中自行选择

## 输出格式

评测结果包含：
- 基本统计（总数、执行成功率、结果匹配率等）
- 配置信息（表数量、列数量、token估算等）
- 详细结果（每个query的评测结果）

## 并发机制

- NL查询生成：使用ThreadPoolExecutor，每个线程独立的OpenAI客户端
- Baseline评测：使用ThreadPoolExecutor，线程安全的客户端管理
- 默认并发数：5（可通过`--max_workers`调整）

## 与论文结果对比

论文中的Baseline结果（不使用TACO-SQL框架）：
- GPT-4o (beijing): 12.06%

目标：通过给足上下文，期望达到接近论文中的Baseline结果。
