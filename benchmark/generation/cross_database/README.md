# 跨数据库JOIN SQL生成项目

## 项目概述

本项目实现了跨数据库JOIN SQL的自动生成，支持2个、3个和4个数据库的跨数据库查询。通过分析数据库表之间的可JOIN关系，生成SQL骨架，然后使用大语言模型填充生成完整的可执行SQL。

## 最终统计结果

### 总体统计
- **总SQL数**: 921个
- **有结果的SQL**: 636个 (69.1%)
- **无结果的SQL**: 285个 (30.9%)

### 按数据库数量分布

| 数据库数量 | 总数 | 有结果 | 成功率 | 目标 | 完成度 | 状态 |
|-----------|------|--------|--------|------|--------|------|
| 2个数据库 | 732 | 585 | 79.9% | 359 | 163.0% | ✅ 已完成 |
| 3个数据库 | 181 | 43 | 23.8% | 105 | 41.0% | ⏳ 进行中 |
| 4个数据库 | 8 | 8 | 100.0% | 2 | 400.0% | ✅ 已完成 |

## 项目结构

### 核心数据文件

```
benchmark/generation/cross_database/
├── joinable_table_pairs.json          # 可JOIN表对数据（101,595个表对）
├── cross_db_skeletons_join.json      # SQL骨架文件（670个骨架）
├── cross_db_graphs_join/             # Schema图文件目录（670个图文件）
└── database_combinations.json        # 数据库组合配置（语义关系）

benchmark/data/beijing/output/
├── cross_db_single_join/              # 生成的SQL文件（输出目录）
└── cross_db_single_join_backup_51/   # 备份目录（有结果的SQL）
```

### 核心脚本

#### 1. 数据准备阶段

**`analyze_joinable_tables.py`** - 分析可JOIN表对
- **功能**: 分析Beijing数据集中的表，找出可以JOIN的表对
- **输入**: 数据库schema文件
- **输出**: `joinable_table_pairs.json`（包含表对、相似度、推荐JOIN列等）
- **使用**: `python3 analyze_joinable_tables.py`

**`2generate_cross_db_skeletons_join.py`** - 生成SQL骨架
- **功能**: 基于可JOIN表对生成JOIN SQL骨架
- **输入**: `joinable_table_pairs.json`
- **输出**: `cross_db_skeletons_join.json`
- **使用**: 
  ```bash
  python3 2generate_cross_db_skeletons_join.py \
    --num_skeletons_2db 500 \
    --num_skeletons_3db 150 \
    --num_skeletons_4db 20
  ```

#### 2. Schema图生成阶段

**`cross_db_1build_schema_graphs.py`** - 生成Schema图
- **功能**: 为SQL骨架生成Schema Linking Graph
- **输入**: `cross_db_skeletons_join.json`、数据库schema
- **输出**: `cross_db_graphs_join/cross_db_graph_*.json`
- **使用**: 
  ```bash
  python3 cross_db_1build_schema_graphs.py \
    --skeleton_file cross_db_skeletons_join.json \
    --output_dir cross_db_graphs_join \
    --database_dir ../../data/beijing/database_chinese
  ```

#### 3. SQL生成阶段

**`cross_db_2fill_sql_placeholders_join.py`** - 核心SQL填充逻辑
- **功能**: 使用LLM填充SQL骨架，生成完整SQL
- **核心函数**: `process_cross_database_skeleton()`
- **特点**: 
  - 支持多数据库JOIN
  - 自动处理ATTACH DATABASE
  - 错误处理和重试机制
  - 保存失败文件用于分析

**`4generate_more_join_sqls_simple.py`** - 批量生成2数据库SQL
- **功能**: 批量生成2个数据库的JOIN SQL
- **特点**: 
  - 自动跳过已生成的SQL
  - 实时进度显示
  - 并发处理
- **使用**: `python3 4generate_more_join_sqls_simple.py`

**`9generate_3db_4db_sqls.py`** - 生成3和4数据库SQL
- **功能**: 专门生成3个和4个数据库的JOIN SQL
- **使用**: 
  ```bash
  # 同时生成3和4数据库
  python3 9generate_3db_4db_sqls.py
  
  # 只生成3数据库
  python3 9generate_3db_4db_sqls.py --only_3db
  
  # 只生成4数据库
  python3 9generate_3db_4db_sqls.py --only_4db
  ```

#### 4. 工具脚本

**`1cleanup_failed_sqls.py`** - 清理无结果的SQL（2数据库）
- **功能**: 删除2个数据库SQL中没有结果的文件
- **使用**: `python3 1cleanup_failed_sqls.py`

**`10cleanup_failed_3db_4db_sqls.py`** - 清理无结果的SQL（3和4数据库）
- **功能**: 删除3和4个数据库SQL中没有结果的文件
- **使用**: 
  ```bash
  # 预览模式
  python3 10cleanup_failed_3db_4db_sqls.py --dry_run
  # 实际清理
  python3 10cleanup_failed_3db_4db_sqls.py
  ```

**`5backup_new_results.py`** - 备份有结果的SQL
- **功能**: 将有结果的SQL备份到备份目录，自动编号
- **使用**: `python3 5backup_new_results.py`

**`7statistics_join_sqls.py`** - 统计SQL生成情况
- **功能**: 统计各数据库数量的SQL分布
- **使用**: `python3 7statistics_join_sqls.py`

**`3check_status.py`** - 检查生成状态
- **功能**: 检查当前生成进度，显示还需要生成多少
- **使用**: `python3 3check_status.py`

## 完整工作流程

### 首次生成（从零开始）

```bash
cd benchmark/generation/cross_database

# 步骤1: 分析可JOIN表对（如果还没有）
python3 analyze_joinable_tables.py

# 步骤2: 生成SQL骨架
python3 2generate_cross_db_skeletons_join.py \
  --num_skeletons_2db 500 \
  --num_skeletons_3db 150 \
  --num_skeletons_4db 20

# 步骤3: 生成Schema图
python3 cross_db_1build_schema_graphs.py

# 步骤4: 生成SQL
python3 4generate_more_join_sqls_simple.py  # 2数据库
python3 9generate_3db_4db_sqls.py          # 3和4数据库

# 步骤5: 备份结果
python3 5backup_new_results.py
```

### 迭代生成（已有部分结果）

```bash
# 步骤1: 清理无结果的SQL
python3 1cleanup_failed_sqls.py
python3 10cleanup_failed_3db_4db_sqls.py

# 步骤2: 备份有结果的SQL
python3 5backup_new_results.py

# 步骤3: 重新生成
python3 4generate_more_join_sqls_simple.py
python3 9generate_3db_4db_sqls.py

# 重复步骤1-3直到达到目标数量
```

## 配置说明

### API配置

配置文件: `benchmark/generation/sql_filling/config.yaml`

```yaml
llm:
  provider: custom
  model: gpt-4o-mini          # 使用gpt-4o-mini避免配额问题
  api_url: https://35.aigcbest.top/v1
  api_key: sk-...
  max_workers: 100            # 并发数
```

### 目标数量配置

在脚本中定义（`4generate_more_join_sqls_simple.py` 和 `9generate_3db_4db_sqls.py`）:

```python
TARGET_COUNTS = {
    2: 359,  # 跨2个数据库
    3: 105,  # 跨3个数据库
    4: 2     # 跨4个数据库
}
```

## 关键技术点

### 1. 可JOIN表对分析

- **方法**: 基于列名相似度匹配（ID、名称、代码等关键词）
- **相似度计算**: 
  - ID匹配: 10分
  - 代码匹配: 8分
  - 名称匹配: 6分
  - 数据类型匹配: +2分
- **阈值**: 相似度 >= 10.0 为高质量表对

### 2. SQL骨架生成

- **2个数据库**: 直接使用表对，生成 `SELECT ... FROM table1 JOIN table2 ON ...`
- **3个数据库**: 链式连接 A-B, B-C
- **4个数据库**: 链式连接 A-B, B-C, C-D
- **多样性**: 
  - 40%使用聚合函数
  - 30%使用ORDER BY
  - 覆盖多个数据库组合

### 3. SQL填充

- **Prompt策略**: 强调JOIN，禁止UNION，鼓励聚合函数和复杂结构
- **执行方式**: 使用SQLite的ATTACH DATABASE功能
- **错误处理**: 保存所有结果（包括失败的），记录错误信息

### 4. 成功率分析

- **2个数据库**: 约80%成功率
- **3个数据库**: 约24%成功率（更复杂）
- **4个数据库**: 约100%成功率（但样本少）

## 数据文件说明

### joinable_table_pairs.json
- **大小**: 73MB
- **内容**: 101,595个可JOIN表对
- **格式**: 
  ```json
  {
    "total_pairs": 101595,
    "joinable_pairs": [
      {
        "db1": "企业服务",
        "db2": "社会保障",
        "table1": "...",
        "table2": "...",
        "column_pairs": [...],
        "best_similarity": 12.0
      }
    ]
  }
  ```

### cross_db_skeletons_join.json
- **大小**: 794KB
- **内容**: 670个SQL骨架
- **格式**: 包含骨架、数据库映射、推荐JOIN列等

### cross_db_graphs_join/
- **内容**: 670个Schema图文件
- **格式**: NetworkX图的JSON序列化

## 常见问题

### 1. API配额错误
- **原因**: 使用了配额受限的模型（如gpt-3.5-turbo）
- **解决**: 使用 `gpt-4o-mini` 模型

### 2. 成功率低
- **原因**: 3个和4个数据库的JOIN更复杂
- **解决**: 多次迭代生成，逐步积累成功结果

### 3. 文件路径错误
- **原因**: 脚本使用了相对路径
- **解决**: 所有脚本已修复为使用绝对路径

### 4. 备份编号混乱
- **原因**: 备份脚本没有正确计算下一个编号
- **解决**: 已修复为从最大连续编号+1开始

## 开发建议

### 继续生成3个数据库的SQL

当前43/105，还需要62个。建议：

```bash
# 清理无结果的
python3 10cleanup_failed_3db_4db_sqls.py --only_3db

# 备份有结果的
python3 5backup_new_results.py

# 重新生成（可能需要多次迭代）
python3 9generate_3db_4db_sqls.py --only_3db --max_workers 5
```

### 优化生成策略

1. **提高3数据库成功率**: 
   - 优化prompt
   - 选择更高质量的表对
   - 简化JOIN条件

2. **增加多样性**:
   - 确保覆盖更多数据库组合
   - 使用不同的SQL结构（聚合、子查询等）

3. **性能优化**:
   - 调整并发数
   - 优化API调用

## 文件清理

运行清理脚本整理目录：

```bash
# 预览模式
python3 11cleanup_and_organize.py --dry_run

# 实际执行
python3 11cleanup_and_organize.py
```

## 快速参考

### 常用命令

```bash
# 生成2数据库SQL
python3 4generate_more_join_sqls_simple.py

# 生成3和4数据库SQL
python3 9generate_3db_4db_sqls.py

# 清理无结果SQL
python3 1cleanup_failed_sqls.py                    # 2数据库
python3 10cleanup_failed_3db_4db_sqls.py          # 3和4数据库

# 备份有结果SQL
python3 5backup_new_results.py

# 查看统计
python3 7statistics_join_sqls.py
python3 3check_status.py                          # 检查生成状态
```

### 迭代生成流程（推荐）

```bash
# 1. 清理无结果的SQL
python3 1cleanup_failed_sqls.py
python3 10cleanup_failed_3db_4db_sqls.py

# 2. 备份有结果的SQL
python3 5backup_new_results.py

# 3. 重新生成
python3 4generate_more_join_sqls_simple.py        # 2数据库
python3 9generate_3db_4db_sqls.py                 # 3和4数据库

# 4. 重复步骤1-3直到达到目标数量
```

## 相关文档

- `README.md` - 本文档（主要快速上手指南）
- `3db4db清理和生成说明.md` - 3和4数据库详细说明
- `项目结构说明.md` - 详细的项目结构说明
- `archive/docs/` - 归档的详细文档和报告（可参考历史信息）
