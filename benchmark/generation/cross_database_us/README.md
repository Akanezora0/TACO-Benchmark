# US数据集跨数据库SQL生成实验框架

## 项目概述

本框架实现了US数据集跨数据库JOIN SQL的自动生成，支持2个、3个和4个数据库的跨数据库查询。参考beijing数据集的实现，适配US数据集（英文）的特点。

## 目标分配

总目标：**6,000条SQL**

| 类型 | 占比 | 目标数量 | 状态 |
|------|------|---------|------|
| 单数据库 | 80.5% | **4,830条** | ✅ 已完成 (当前: 5,715条) |
| 跨2个数据库 | 15.0% | **900条** | ⏳ 待生成 |
| 跨3个数据库 | 4.4% | **264条** | ⏳ 待生成 |
| 跨4个数据库 | 0.1% | **6条** | ⏳ 待生成 |

## 项目结构

```
benchmark/generation/cross_database_us/
├── README.md                              # 本文档
├── 0check_status.py                       # 检查生成状态
├── 1analyze_joinable_tables.py           # 分析可JOIN表对
├── 2generate_cross_db_skeletons_join.py   # 生成SQL骨架
├── 3build_schema_graphs.py                # 生成Schema图
├── 4generate_2db_sqls.py                 # 生成2数据库SQL
├── 5generate_3db_4db_sqls.py             # 生成3和4数据库SQL
├── 6cleanup_failed_sqls.py               # 清理无结果SQL
├── 7backup_results.py                    # 备份有结果SQL
├── 8statistics.py                        # 统计生成情况
└── run_all.py                            # 一键运行所有步骤

benchmark/data/us/output/
├── cross_db_single_join/                  # 生成的SQL文件（输出目录）
└── cross_db_single_join_backup/          # 备份目录（有结果的SQL）
```

## 快速开始

### 一键生成（推荐）

```bash
cd /home/u2023103807/TACO/benchmark/generation/cross_database_us

# 查看当前状态
python3 0check_status.py

# 一键生成所有跨数据库SQL
python3 run_all.py

# 或者分步执行
python3 run_all.py --step 1  # 只执行步骤1（分析可JOIN表对）
python3 run_all.py --step 2  # 只执行步骤2（生成骨架）
# ...
```

### 分步执行

```bash
# 步骤1: 分析可JOIN表对
python3 1analyze_joinable_tables.py

# 步骤2: 生成SQL骨架
python3 2generate_cross_db_skeletons_join.py

# 步骤3: 生成Schema图
python3 3build_schema_graphs.py

# 步骤4: 生成2数据库SQL
python3 4generate_2db_sqls.py

# 步骤5: 生成3和4数据库SQL
python3 5generate_3db_4db_sqls.py
```

## 核心脚本说明

### 0check_status.py - 检查状态

```bash
python3 0check_status.py
```

功能：
- 显示当前各类型SQL的生成情况
- 显示还需要生成多少
- 显示成功率统计

### 1analyze_joinable_tables.py - 分析可JOIN表对

```bash
python3 1analyze_joinable_tables.py
```

功能：
- 分析US数据集中的表，找出可以JOIN的表对
- 输出：`joinable_table_pairs.json`

### 2generate_cross_db_skeletons_join.py - 生成SQL骨架

```bash
python3 2generate_cross_db_skeletons_join.py \
    --num_skeletons_2db 1200 \
    --num_skeletons_3db 400 \
    --num_skeletons_4db 20
```

功能：
- 基于可JOIN表对生成JOIN SQL骨架
- 输出：`cross_db_skeletons_join.json`

### 3build_schema_graphs.py - 生成Schema图

```bash
python3 3build_schema_graphs.py
```

功能：
- 为SQL骨架生成Schema Linking Graph
- 输出：`cross_db_graphs_join/` 目录

### 4generate_2db_sqls.py - 生成2数据库SQL

```bash
python3 4generate_2db_sqls.py
```

功能：
- 批量生成2个数据库的JOIN SQL
- 自动跳过已生成的SQL
- 实时进度显示

### 5generate_3db_4db_sqls.py - 生成3和4数据库SQL

```bash
# 同时生成3和4数据库
python3 5generate_3db_4db_sqls.py

# 只生成3数据库
python3 5generate_3db_4db_sqls.py --only_3db

# 只生成4数据库
python3 5generate_3db_4db_sqls.py --only_4db
```

## 工具脚本

### 6cleanup_failed_sqls.py - 清理无结果SQL

```bash
# 预览模式
python3 6cleanup_failed_sqls.py --dry_run

# 实际清理
python3 6cleanup_failed_sqls.py
```

### 7backup_results.py - 备份有结果SQL

```bash
python3 7backup_results.py
```

### 8statistics.py - 统计生成情况

```bash
python3 8statistics.py
```

## 迭代生成流程（推荐）

```bash
# 1. 清理无结果的SQL
python3 6cleanup_failed_sqls.py

# 2. 备份有结果的SQL
python3 7backup_results.py

# 3. 重新生成
python3 4generate_2db_sqls.py
python3 5generate_3db_4db_sqls.py

# 4. 查看统计
python3 8statistics.py

# 5. 重复步骤1-4直到达到目标数量
```

## 配置说明

### API配置

配置文件: `benchmark/generation/sql_filling/config.yaml`

```yaml
llm:
  provider: custom
  model: gpt-4o-mini
  api_url: https://35.aigcbest.top/v1
  api_key: sk-...
  max_workers: 100
```

### 目标数量配置

在脚本中定义：

```python
TARGET_COUNTS = {
    2: 900,  # 跨2个数据库
    3: 264,  # 跨3个数据库
    4: 6     # 跨4个数据库
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

- **Prompt策略**: 英文prompt，强调JOIN，禁止UNION，鼓励聚合函数
- **执行方式**: 使用SQLite的ATTACH DATABASE功能
- **错误处理**: 保存所有结果（包括失败的），记录错误信息

## 常见问题

### 1. API配额错误
- **解决**: 使用 `gpt-4o-mini` 模型

### 2. 成功率低
- **原因**: 3个和4个数据库的JOIN更复杂
- **解决**: 多次迭代生成，逐步积累成功结果

### 3. 文件路径错误
- **解决**: 所有脚本使用绝对路径

## 开发建议

### 继续生成策略

1. **提高成功率**: 
   - 优化prompt
   - 选择更高质量的表对
   - 简化JOIN条件

2. **增加多样性**:
   - 确保覆盖更多数据库组合
   - 使用不同的SQL结构（聚合、子查询等）

3. **性能优化**:
   - 调整并发数
   - 优化API调用

