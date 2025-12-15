# SQL骨架生成模块

## 概述

为beijing数据集的每个数据库生成高质量的SQL骨架，包含JOIN、子查询、聚合函数等复杂结构。

## 主要脚本

- **`generate_for_databases_improved.py`** - 主要脚本（改进版）
  - 结合旧数据库CFG规则和新数据库专家例子
  - 为每个数据库生成不同的结果（使用数据库名称作为随机种子）
  - 支持高级转换策略（旋转、剪枝、合并等）
  - 改进SQL骨架生成逻辑，支持JOIN、子查询、聚合等复杂结构
  - 增强验证逻辑，修复语法错误
  - 支持默认路径，可直接运行

## 快速开始

### 方式1: 使用默认路径（推荐）

```bash
cd benchmark/generation/sql_skeleton_generation
python generate_for_databases_improved.py
```

或使用shell脚本：

```bash
cd benchmark/generation/sql_skeleton_generation
./run.sh
```

### 方式2: 自定义参数

```bash
cd benchmark/generation/sql_skeleton_generation
python generate_for_databases_improved.py \
    --num_samples 100 \
    --total_skeletons 200
```

## 参数说明

所有参数都有默认值，可以直接运行：

- `--database_dir`: 数据库目录路径（默认：`benchmark/data/beijing/database`）
- `--expert_file`: 专家例子文件路径（默认：`benchmark/data/target/expert_skeletons_beijing.json`）
- `--old_cfg_file`: 旧数据库CFG文件路径（默认：`old/saturn/TACO-Benchmark-all/beijing/data/old_ast_cfg.json`）
- `--output_dir`: 输出目录路径（默认：`benchmark/data/beijing/output`）
- `--old_data_file`: 旧数据文件（默认：`old/saturn/TACO-Benchmark-all/beijing/data/xcity_sql_skeletons.json`）
- `--new_logs_file`: 新日志文件（默认：`benchmark/data/target/expert_skeletons_beijing.json`）
- `--num_samples`: 每个数据库生成的结构数量（默认100）
- `--total_skeletons`: 每个数据库生成的骨架总数（默认200）

## 输出结构

所有输出保存在 `benchmark/data/beijing/output/`：

```
output/
├── ast_cfg/              # 26个CFG文件（每个数据库51条）
├── sql_structure/        # 26个结构文件（每个数据库100个）
└── sql_skeleton/         # 26个骨架文件（每个数据库200个）
```

## 关键改进

1. **完善专家例子**：从20条增加到41条，包含13个JOIN、12个子查询、8个聚合函数
2. **结合旧数据库**：使用xcity的SQL骨架（226条，包含37个JOIN、27个子查询）
3. **改进验证逻辑**：修复语法错误检查，确保生成的SQL骨架有效
4. **数据库差异化**：每个数据库使用不同的随机种子，确保结果不同
5. **支持默认路径**：所有参数都有默认值，可直接运行

## 结果质量

### 安全生产数据库（示例）

| 指标 | 原始结果 | 新结果 | 改进 |
|------|---------|--------|------|
| 唯一骨架数量 | 117 | 172 | ✅ +47% |
| 包含JOIN | 28 | 47 | ✅ +68% |
| 包含子查询 | 16 | 26 | ✅ +63% |
| 包含括号 | 35 | 54 | ✅ +54% |
| 语法错误 | 0 | 0 | ✅ |

## 可重复性

- 使用数据库名称的哈希值作为随机种子，确保结果可复现
- 所有路径使用绝对路径，避免相对路径问题
- 输入文件相同的情况下，输出结果应该一致

## 文件说明

- `generate_for_databases_improved.py` - 主要脚本（改进版）
- `generate_for_databases.py` - 旧版本（保留作为参考）
- `run.sh` - 快速运行脚本
- `README.md` - 本文件
- `使用说明.md` - 详细使用说明
- `目录整理总结.md` - 目录整理说明
