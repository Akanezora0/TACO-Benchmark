# TACO-SQL实验核心设置与Prompt策略

本文档详细说明TACO-SQL框架的实验设置、Prompt策略和核心实现思路，用于论文开源展示。

## 一、实验设置（Experimental Settings）

### 1.1 消融实验设计（Ablation Study）

TACO-SQL框架采用逐步添加组件的消融实验设计，评估每个组件的贡献：

| 实验设置 | 组件配置 | 说明 |
|---------|---------|------|
| **Origin** | 原始查询 + 完整Schema | Baseline设置，不使用任何TACO-SQL组件 |
| **QR** | + Question Rewriting | 仅使用问题转写组件 |
| **QR+TL** | + Question Rewriting + Table Linking | 使用问题转写和表格检索 |
| **QR+TL+QP** | + Question Rewriting + Table Linking + Query Planning | 完整TACO-SQL框架 |

### 1.2 评测模型类型

实验涵盖以下四类Text-to-SQL模型：

#### Base LLMs
- GPT-4, GPT-4o, GPT-4o-mini, GPT-o1
- DeepSeek-v2.5, DeepSeek-R1
- Llama3-70b, Qwen2-72b

#### LLM-Based方法
- DIN-SQL
- MAC-SQL

#### SFT-Based方法
- CodeS-33B, CodeS-15B
- Qwen2.5-Coder-32B
- Deepseek-coder-6.7b

#### Hybrid方法
- CHESS
- Zero-NL2SQL
- DIAL-SQL

### 1.3 评估指标

**主要指标：Execution Accuracy (EX)**
- 定义：预测SQL执行结果与标准答案完全匹配的比例
- 计算公式：`EX = (正确执行的查询数) / (总查询数)`

**辅助指标：**
- Recall@K：表检索准确率（K=1,3,5,10）
- SQL语法正确率
- 执行时间

## 二、Prompt策略（Prompt Strategies）

### 2.1 Question Rewriting（问题转写）

#### 功能目标
- 澄清模糊或冗余的用户查询
- 规范化查询表达，去除冗余信息
- 明确查询意图

#### Prompt模板

**System Prompt**：
```
You rewrite user questions for SQL retrieval. Remove irrelevant chatter, disambiguate entities, 
and output one concise sentence expressing the core intent while preserving key filters.
```

**Few-Shot Examples**：
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

**Prompt构建方式**：
- System message + Few-shot examples (user-assistant pairs) + Current query

#### 实现方式
- **方法**：基于LLM的few-shot prompting
- **模型**：GPT-4o / DeepSeek等通用大语言模型
- **参数设置**：
  - Temperature: 0.3（较低温度保证转写稳定性）
  - Top-p: 0.9
  - Max Tokens: 512

#### 示例

**输入（原始查询）**：
```
我想看看最近几年北京地区企业注册的情况，包括注册数量和注册资本
```

**输出（转写后）**：
```
查询北京地区企业注册数据：注册数量、注册资本，按年份统计
```

### 2.2 Table Linking（表格检索）

#### 功能目标
- 从大规模异构数据库中识别相关表
- 减少Schema规模，提高SQL生成效率

#### 实现架构

**离线训练阶段：**
- **模型**：SBERT双塔模型（Sentence-BERT）
- **基础模型**：`paraphrase-multilingual-MiniLM-L12-v2`
- **训练方法**：对比学习（Multiple Negatives Ranking Loss）
- **训练数据**：query-table对（正样本）
- **训练参数**：
  - Epochs: 3
  - Batch Size: 16
  - Learning Rate: 2e-5
  - Warmup Steps: 100

**在线检索阶段：**
- **方法**：语义检索（余弦相似度）
- **检索策略**：Top-K检索（默认K=5）
- **表信息构建**：`表描述 + 列名列表`

#### 表信息提取

```python
def extract_table_info(row):
    description = row['表的描述']
    column_names = [col for col in row.index if 'column_content' in col and pd.notnull(row[col])]
    table_info = description + ' ' + ' '.join(column_names)
    return table_info
```

**表信息格式**：`表描述 + 列名列表`（空格分隔）

#### 训练数据构建

```python
# 从训练集中构建query-table对
for query, table_names in train_data.items():
    for table_name in table_names:
        table_info = extract_table_info(merged_table[table_name])
        # 构建正样本对
        train_examples.append(InputExample(
            texts=[query, table_info], 
            label=1.0
        ))
```

#### 检索流程

```python
# 1. 编码查询和所有表
query_embeddings = query_model.encode(queries, convert_to_tensor=True)
table_embeddings = table_model.encode(table_infos, convert_to_tensor=True)

# 2. 语义检索（余弦相似度）
hits = util.semantic_search(query_embedding, table_embeddings, top_k=k)

# 3. 返回Top-K表名
retrieved_tables = [merged_table.iloc[hit['corpus_id']]['table_name'] for hit in hits]
```

#### 评估指标
- Recall@1, Recall@3, Recall@5, Recall@10
- 计算方式：检索到的表中是否包含真实相关表

### 2.3 Query Planning（查询规划）

#### 功能目标
- 将复杂查询拆解为多个简单子查询
- 确定执行顺序和依赖关系
- 处理跨数据库查询

#### Prompt模板

```python
"""请将以下查询拆解为多个简单的子查询，并确定执行顺序。

原始查询：{query}

相关表：{relevant_tables}

Schema信息：{schema_info}

请以JSON格式输出执行计划，格式如下：
[
    {
        "subquery": "子查询描述",
        "tables": ["表1", "表2"],
        "order": 1,
        "dependencies": []
    },
    ...
]

执行计划："""
```

#### 实现方式
- **方法**：基于LLM的结构化规划
- **模型**：GPT-4o
- **参数设置**：
  - Temperature: 0.3（较低温度保证规划稳定性）
  - Max Tokens: 1024
- **输出格式**：JSON结构化计划

#### 执行计划结构

```json
[
    {
        "subquery": "查询企业注册基本信息",
        "tables": ["企业注册表"],
        "order": 1,
        "dependencies": []
    },
    {
        "subquery": "按年份统计注册数量和注册资本",
        "tables": ["企业注册表"],
        "order": 2,
        "dependencies": [1]
    }
]
```

### 2.4 SQL Generation（SQL生成）

#### 2.4.1 Baseline Prompt（原始设置）

**适用场景**：Origin实验设置（原始查询 + 完整Schema）

**Prompt模板**：

```python
"""你是一个SQL专家。根据自然语言查询和数据库Schema，生成对应的SQL查询语句。

{schema_text}

自然语言查询：{query}

要求：
1. 生成完整、可执行的SQL语句
2. 所有表名和列名必须用双引号包裹（包括中文和特殊字符）
3. 确保SQL语法正确，可以在SQLite上执行
4. 只输出SQL语句，不要添加任何解释或注释

数据库：{database}

SQL查询："""
```

**关键设置**：
- Schema包含：所有表（根据模型上下文窗口决定）
- 表名和列名：使用双引号包裹（支持中文）
- 输出格式：仅SQL语句，无额外解释

#### 2.4.2 TACO-SQL Prompt（使用Table Linking后）

**适用场景**：QR+TL和QR+TL+QP实验设置

**Prompt模板**：

```python
"""你是一个SQL专家。根据自然语言查询和相关数据库Schema，生成对应的SQL查询语句。

相关表Schema信息：
{filtered_schema_text}

自然语言查询：{rewritten_query}

要求：
1. 生成完整、可执行的SQL语句
2. 所有表名和列名必须用双引号包裹（包括中文和特殊字符）
3. 确保SQL语法正确，可以在SQLite上执行
4. 只输出SQL语句，不要添加任何解释或注释
5. 仅使用上述相关表，不要使用未列出的表

数据库：{database}

SQL查询："""
```

**关键改进**：
- Schema过滤：仅包含Table Linking检索到的相关表（Top-K）
- 查询输入：使用Question Rewriting后的查询
- 表限制：明确要求仅使用相关表

#### 2.4.3 模型特定配置

**Base LLM配置**（GPT-4o等）：
```python
{
    "temperature": 0.1,  # 低温度保证SQL准确性
    "max_tokens": 2000,
    "context_window": 128000  # GPT-4o大上下文窗口
}
```

**SFT模型配置**（CodeS等）：
```python
{
    "max_new_tokens": 512,
    "num_beams": 4,  # Beam search提高生成质量
    "num_return_sequences": 4  # 生成多个候选
}
```

**Schema过滤配置**（SFT模型使用）：
```python
{
    "enabled": true,
    "model_path": "sic_ckpts/sic_spider",  # Schema Item Classifier
    "max_tables": 7,
    "max_columns": 20
}
```

**Schema过滤流程**（SFT模型）：
1. 使用Schema Item Classifier (SIC)对表/列进行相关性评分
2. 选择Top-K表（默认7个）和每表Top-M列（默认20个）
3. 构建过滤后的Schema序列
4. 输入格式：`schema_sequence + content_sequence + query`

**SFT模型输入格式**：
```python
prefix_seq = (
    data["schema_sequence"] + "\n" + 
    data["content_sequence"] + "\n" + 
    data["text"] + "\n"
)
```

## 三、实验代码思路（Code Implementation Strategy）

### 3.1 Pipeline架构

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
            result['relevant_tables'] = []  # 使用完整Schema
        
        # Step 3: Query Planning
        if 'qp' in enable_components and result['relevant_tables']:
            result['execution_plan'] = self.query_planner.plan(
                result['rewritten_query'],
                result['relevant_tables']
            )
        else:
            # 默认单步计划
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

### 3.2 实验设置映射

| 实验设置 | enable_components | Schema策略 |
|---------|------------------|-----------|
| Origin | `[]` | 完整Schema |
| QR | `['qr']` | 完整Schema |
| QR+TL | `['qr', 'tl']` | 过滤Schema（Top-K表） |
| QR+TL+QP | `['qr', 'tl', 'qp']` | 过滤Schema + 查询规划 |

### 3.3 Schema格式化策略

#### 完整Schema（Origin, QR设置）

```python
def format_schema_simple(schema: Dict, max_tables: int = None) -> str:
    """格式化完整Schema"""
    text = "数据库Schema信息：\n\n"
    
    for table in schema.get('tables', []):
        table_name = table.get('table_name', '')
        columns = table.get('columns', [])
        
        text += f"表：{table_name}\n"
        text += "  列：\n"
        for col in columns:
            col_name = col.get('column_name', '')
            col_type = col.get('data_type', 'TEXT')
            text += f"    - {col_name} ({col_type})\n"
        text += "\n"
    
    return text
```

#### 过滤Schema（QR+TL, QR+TL+QP设置）

```python
def format_schema_filtered(schema: Dict, relevant_tables: List[str]) -> str:
    """格式化过滤后的Schema（仅包含相关表）"""
    text = "相关表Schema信息：\n\n"
    
    for table in schema.get('tables', []):
        table_name = table.get('table_name', '')
        if table_name not in relevant_tables:
            continue
        
        # ... 格式化表信息 ...
    
    return text
```

## 四、核心设置总结

### 4.1 组件配置参数

| 组件 | 关键参数 | 默认值 | 说明 |
|------|---------|--------|------|
| Question Rewriting | temperature | 0.7 | 平衡创造性和准确性 |
| Question Rewriting | max_tokens | 512 | 转写查询长度限制 |
| Table Linking | top_k | 5 | 检索相关表数量 |
| Table Linking | training_epochs | 3 | SBERT训练轮数 |
| Query Planning | temperature | 0.3 | 低温度保证规划稳定性 |
| Query Planning | max_tokens | 1024 | 规划输出长度 |
| SQL Generation | temperature | 0.1 | 低温度保证SQL准确性 |
| SQL Generation | max_tokens | 2000 | SQL生成长度限制 |

### 4.2 实验运行流程

```bash
# 1. Baseline实验（Origin设置）
python experiments/baselines/base_llm/evaluate_baseline.py \
    --model gpt-4o \
    --setting origin \
    --dataset taco_beijing

# 2. TACO-SQL消融实验
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr_tl_qp \
    --model gpt-4o \
    --dataset taco_beijing

# 3. 评估结果
python experiments/evaluation/exec_eval.py \
    --pred results/raw/predictions.json \
    --gold data/test.json
```

### 4.3 关键设计决策

1. **Schema过滤策略**
   - **问题**：完整Schema包含数千张表，超出模型上下文窗口
   - **解决方案**：使用Table Linking检索Top-K表（默认K=5）
   - **效果**：减少上下文长度90%+，提高模型注意力集中度

2. **查询转写必要性**
   - **问题**：真实用户查询包含冗余、模糊表达
   - **解决方案**：使用Few-shot LLM进行查询转写
   - **效果**：规范化表达，提高Table Linking和SQL生成准确性

3. **查询规划适用场景**
   - **问题**：跨数据库复杂查询需要多步执行
   - **解决方案**：LLM生成结构化执行计划
   - **效果**：将复杂查询拆解为简单子查询，逐步执行

4. **Prompt设计原则**
   - **明确输出格式**：仅SQL语句，无额外解释（避免解析错误）
   - **表名/列名处理**：双引号包裹（支持中文和特殊字符）
   - **Schema结构清晰**：表-列层次结构，便于模型理解
   - **Few-shot示例**：提供高质量示例，引导模型行为

5. **温度参数选择**
   - **Question Rewriting**: 0.3（平衡创造性和准确性）
   - **Query Planning**: 0.3（保证规划稳定性）
   - **SQL Generation**: 0.1（保证SQL准确性，避免语法错误）

## 五、文件结构说明

### 5.1 核心代码位置

```
experiments/
├── baselines/                    # 基线实验
│   └── base_llm/
│       └── evaluate_baseline.py  # Baseline评估代码（包含Origin设置）
├── taco_sql_exp/                 # TACO-SQL消融实验
│   ├── qr/                       # + Question Rewriting
│   ├── qr_tl/                    # + QR + Table Linking
│   └── qr_tl_qp/                 # 完整TACO-SQL
├── evaluation/                    # 评估工具
│   ├── exec_eval.py              # 执行准确率评估
│   └── evaluation.py             # 综合评估
└── results/                      # 实验结果
```

### 5.2 TACO-SQL组件代码位置

```
taco_sql/
├── question_rewriting/
│   └── rewriting.py               # Question Rewriting实现
├── table_linking/
│   ├── training/
│   │   └── finetune_sbert.py     # SBERT训练代码
│   └── retrieval/
│       └── table_retrieval.py    # 表格检索实现
├── query_planning/
│   └── planning.py                # 查询规划实现
├── sql_generation/
│   └── text2sql.py                # SQL生成实现
└── pipeline/
    └── taco_sql_pipeline.py      # 完整Pipeline
```

## 六、实验复现说明

### 6.1 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API密钥（Base LLM）
export OPENAI_API_KEY="your_api_key"
export DEEPSEEK_API_KEY="your_api_key"
```

### 6.2 数据准备

```bash
# 1. 准备测试数据集
# 位置：benchmark/data/final/test.json

# 2. 准备Schema文件
# 位置：benchmark/data/schemas/

# 3. 准备Table Linking模型（如已训练）
# 位置：taco_sql/table_linking/models/
# 或运行训练脚本：
python taco_sql/table_linking/training/finetune_sbert.py
```

### 6.3 运行实验

#### 方式1：使用Pipeline（推荐）

```python
# 示例：运行完整TACO-SQL实验
from taco_sql.pipeline.taco_sql_pipeline import TACOSQLPipeline
import yaml

# 加载配置
with open('configs/taco_sql_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建Pipeline
pipeline = TACOSQLPipeline(config=config)

# 运行实验
result = pipeline.run(
    user_query="查询北京地区企业注册情况",
    enable_components=['qr', 'tl', 'qp', 'sg']
)

print(f"生成的SQL: {result['sql_queries']}")
```

#### 方式2：使用实验脚本

```bash
# Baseline实验（Origin设置）
python experiments/baselines/base_llm/evaluate_baseline.py \
    --model gpt-4o \
    --dataset taco_beijing \
    --output results/baseline_gpt4o.json

# TACO-SQL消融实验
python experiments/taco_sql_exp/run_ablation.py \
    --setting qr_tl_qp \
    --model gpt-4o \
    --dataset taco_beijing \
    --output results/taco_sql_gpt4o.json
```

### 6.4 评估结果

```bash
# 执行准确率评估
python experiments/evaluation/exec_eval.py \
    --pred results/predictions.json \
    --gold benchmark/data/final/test.json \
    --output results/evaluation_report.json

# 结果对比分析
python experiments/evaluation/compare.py \
    --results_dir results/ \
    --output results/comparison_report.md
```

---

**文档版本**：v1.0  
**最后更新**：2025年1月  
**维护者**：TACO项目组

