# SQL骨架内容填充 - 改进版

## 概述

本模块实现了改进的SQL骨架内容填充功能，主要改进点：

1. **真正利用图结构**：使用SQL-Schema Linking Graph来选择相关的表和列
2. **利用外键关系**：优先选择有外键关系的表对进行JOIN操作
3. **增强Prompt**：包含表描述、列信息、外键关系等详细信息
4. **智能推理**：根据SQL骨架的语义（JOIN、聚合、子查询等）选择最合适的表

## 文件结构

```
sql_filling/
├── 1build_schema_graphs_improved.py  # 构建SQL-Schema Linking Graph（改进版）
├── 2fill_sql_placeholders_improved.py # 填充SQL占位符（改进版）
├── config.yaml                        # 配置文件
├── README.md                          # 本文件
└── 之前实现方式分析.md                 # 之前实现方式的分析文档
```

## 使用流程

### 步骤1：构建SQL-Schema Linking Graph

```bash
cd /home/u2023103807/TACO/benchmark/generation/sql_filling
python3 1build_schema_graphs_improved.py
```

或者使用自定义参数：

```bash
python3 1build_schema_graphs_improved.py \
    --skeleton_dir ../../data/beijing/output/sql_skeleton \
    --database_dir ../../data/beijing/database \
    --output_dir ../../data/beijing/output/graph
```

**输出**：
- `output_dir/{数据库名}/{数据库名}_graph_{索引}.graphml` - 图文件
- `output_dir/{数据库名}/{数据库名}_metadata_{索引}.json` - 图元数据

### 步骤2：填充SQL占位符

```bash
python3 2fill_sql_placeholders_improved.py
```

或者使用自定义参数：

```bash
python3 2fill_sql_placeholders_improved.py \
    --skeleton_dir ../../data/beijing/output/sql_skeleton \
    --database_dir ../../data/beijing/database \
    --graph_dir ../../data/beijing/output/graph \
    --output_dir ../../data/beijing/output \
    --max_retries 3
```

**输出**：
- `output_dir/single/{数据库名}/generated_sql_{索引}.json` - 生成的SQL语句

## 关键改进

### 1. 图结构利用

**之前**：构建了图但没有使用
**现在**：
- 使用图的连通性来选择相关的表和列
- 利用外键边来指导JOIN操作
- 使用图元数据来增强Prompt

### 2. 智能表选择

**之前**：完全随机选择表
**现在**：
- 如果SQL骨架包含JOIN，优先选择有外键关系的表对
- 如果没有外键关系，回退到随机选择（保留原有逻辑）
- 根据SQL骨架的语义需求选择合适数量的表

### 3. 增强Prompt

**之前**：只提供表名和列名列表
**现在**：
- 表详细信息（描述、注释）
- 列信息（数据类型）
- 外键关系说明
- SQL骨架分析提示（JOIN、聚合、子查询等）

### 4. 智能推理

**之前**：没有推理机制
**现在**：
- 分析SQL骨架的语义（是否有JOIN、聚合、子查询）
- 根据语义选择最合适的表
- 提供针对性的提示信息

## 配置说明

配置文件：`config.yaml`

```yaml
llm:
  provider: "custom"
  model: "gpt-4o"
  temperature: 0.1
  max_tokens: 8000
  api_url: "https://35.aigcbest.top/v1"
  api_key: "your-api-key"

processing:
  max_retries: 3
  timeout: 30
```

## 输出格式

生成的SQL文件格式：

```json
{
  "sql": "SELECT \"表名\".\"列名\" FROM \"表名\" WHERE \"表名\".\"列名\" = 'value';",
  "results": [[...], [...]],
  "sql_skeleton": "SELECT _ FROM _ WHERE _ = _",
  "database": "数据库名",
  "tables": {
    "表名": ["表名.列1", "表名.列2"]
  },
  "metadata": {
    "has_join": false,
    "has_subquery": false,
    "has_aggregate": false
  }
}
```

## 注意事项

1. **图文件必须先构建**：在填充SQL之前，必须先运行步骤1构建图文件
2. **API配置**：确保`config.yaml`中的API配置正确
3. **数据库文件**：确保数据库文件（.db）存在于database目录中
4. **重试机制**：如果生成失败，会自动重试（最多3次）

## 故障排查

1. **图文件不存在**：运行步骤1构建图文件
2. **API调用失败**：检查API配置和网络连接
3. **SQL执行失败**：检查数据库文件是否存在，SQL语法是否正确
4. **没有外键关系**：如果没有外键关系，会自动回退到随机选择表

## 与之前实现的对比

| 特性 | 之前实现 | 改进版 |
|------|---------|--------|
| 图结构利用 | ❌ 构建了但没用 | ✅ 真正利用图结构 |
| 外键关系 | ❌ 没有利用 | ✅ 优先选择有外键关系的表 |
| Prompt信息 | ❌ 只有表名列名 | ✅ 包含表描述、列信息、外键关系 |
| 智能推理 | ❌ 没有 | ✅ 根据SQL语义选择表 |
| 表选择策略 | ❌ 完全随机 | ✅ 智能选择（有外键优先） |

## 下一步改进方向

1. 支持跨数据库查询
2. 更智能的表选择算法（考虑语义相似度）
3. 更完善的错误处理和恢复机制
4. 支持更多SQL语法（UNION、CASE等）

