# 跨数据库JOIN SQL生成实验 - 最终版README

## 实验概述

本实验实现了跨数据库JOIN SQL的自动生成，支持2个、3个和4个数据库的跨数据库查询。

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

## 核心文件结构

### 数据文件
```
benchmark/generation/cross_database/
├── joinable_table_pairs.json          # 可JOIN表对数据（101,595个表对）
├── cross_db_skeletons_join.json       # SQL骨架文件（670个骨架）
├── cross_db_graphs_join/              # Schema图文件目录（670个图文件）
└── database_combinations.json        # 数据库组合配置

benchmark/data/beijing/output/
├── cross_db_single_join/              # 生成的SQL文件（输出目录）
└── cross_db_single_join_backup_51/   # 备份目录（有结果的SQL）
```

### 核心脚本

#### 1. 数据准备
- `analyze_joinable_tables.py` - 分析可JOIN表对，生成 `joinable_table_pairs.json`
- `2generate_cross_db_skeletons_join.py` - 生成JOIN SQL骨架

#### 2. Schema图生成
- `cross_db_1build_schema_graphs.py` - 为SQL骨架生成Schema图

#### 3. SQL生成
- `cross_db_2fill_sql_placeholders_join.py` - 核心SQL填充逻辑（使用LLM）
- `4generate_more_join_sqls_simple.py` - 批量生成2数据库SQL
- `9generate_3db_4db_sqls.py` - 生成3和4数据库SQL

#### 4. 工具脚本
- `1cleanup_failed_sqls.py` - 清理无结果的SQL（2数据库）
- `10cleanup_failed_3db_4db_sqls.py` - 清理无结果的SQL（3和4数据库）
- `5backup_new_results.py` - 备份有结果的SQL
- `7statistics_join_sqls.py` - 统计SQL生成情况
- `3check_status.py` - 检查生成状态

## 快速开始

### 生成2个数据库的SQL

```bash
cd benchmark/generation/cross_database
python3 4generate_more_join_sqls_simple.py
```

### 生成3个和4个数据库的SQL

```bash
# 同时生成3和4数据库
python3 9generate_3db_4db_sqls.py

# 只生成3数据库
python3 9generate_3db_4db_sqls.py --only_3db

# 只生成4数据库
python3 9generate_3db_4db_sqls.py --only_4db
```

### 清理无结果的SQL

```bash
# 清理2数据库的无结果SQL
python3 1cleanup_failed_sqls.py

# 清理3和4数据库的无结果SQL
python3 10cleanup_failed_3db_4db_sqls.py

# 预览模式（推荐先预览）
python3 10cleanup_failed_3db_4db_sqls.py --dry_run
```

### 备份有结果的SQL

```bash
python3 5backup_new_results.py
```

### 查看统计

```bash
python3 7statistics_join_sqls.py
```

## 完整工作流程

### 首次生成（从零开始）

1. **分析可JOIN表对**（如果还没有）:
   ```bash
   python3 analyze_joinable_tables.py
   ```

2. **生成SQL骨架**:
   ```bash
   python3 2generate_cross_db_skeletons_join.py --num_skeletons_2db 500 --num_skeletons_3db 150 --num_skeletons_4db 20
   ```

3. **生成Schema图**:
   ```bash
   python3 cross_db_1build_schema_graphs.py
   ```

4. **生成SQL**:
   ```bash
   # 生成2数据库SQL
   python3 4generate_more_join_sqls_simple.py
   
   # 生成3和4数据库SQL
   python3 9generate_3db_4db_sqls.py
   ```

5. **备份结果**:
   ```bash
   python3 5backup_new_results.py
   ```

### 迭代生成（已有部分结果）

1. **清理无结果的SQL**:
   ```bash
   python3 1cleanup_failed_sqls.py
   python3 10cleanup_failed_3db_4db_sqls.py
   ```

2. **备份有结果的SQL**:
   ```bash
   python3 5backup_new_results.py
   ```

3. **重新生成**:
   ```bash
   python3 4generate_more_join_sqls_simple.py
   python3 9generate_3db_4db_sqls.py
   ```

4. **重复步骤1-3直到达到目标数量**

## 配置说明

### API配置

API配置在 `benchmark/generation/sql_filling/config.yaml`:
```yaml
llm:
  provider: custom
  model: gpt-4o-mini
  api_url: https://35.aigcbest.top/v1
  api_key: sk-...
  max_workers: 100
```

### 目标数量

在 `4generate_more_join_sqls_simple.py` 和 `9generate_3db_4db_sqls.py` 中定义:
```python
TARGET_COUNTS = {
    2: 359,  # 跨2个数据库
    3: 105,  # 跨3个数据库
    4: 2     # 跨4个数据库
}
```

## 目录清理

运行清理脚本整理目录：

```bash
# 预览模式（推荐先预览）
python3 11cleanup_and_organize.py --dry_run

# 实际执行清理
python3 11cleanup_and_organize.py
```

清理脚本会：
- 将旧版本脚本移到 `archive/scripts/`
- 将过时文档移到 `archive/docs/`
- 删除所有日志文件
- 删除临时数据文件

## 重要说明

1. **成功率**: 
   - 2个数据库: 约80%
   - 3个数据库: 约24%
   - 4个数据库: 约100%（但样本少）

2. **迭代生成**: 由于成功率不是100%，需要多次迭代生成才能达到目标数量

3. **备份**: 每次生成后建议立即备份有结果的SQL

4. **清理**: 定期清理无结果的SQL，释放空间

## 相关文档

- `实验总结和目录结构.md` - 详细的实验总结和目录结构说明
- `生成情况统计报告.md` - 详细的统计报告
- `3db4db清理和生成说明.md` - 3和4数据库的详细说明

