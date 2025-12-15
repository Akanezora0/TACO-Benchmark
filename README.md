# TACO: A Benchmark for Open-Domain Text-to-SQL

## 项目概述

TACO是一个针对开放域场景的Text-to-SQL基准测试数据集和框架。本项目包含：

1. **Benchmark数据构造**: 从原始数据构建Text-to-SQL benchmark
2. **TACO-SQL框架**: 模块化的Text-to-SQL解决方案
3. **实验框架**: 评估各种SOTA模型和方法

## 项目结构

```
TACO/
├── benchmark/              # Benchmark数据构造
│   ├── data_collection/   # 原始数据收集
│   ├── sql_generation/     # SQL生成（骨架→填充）
│   ├── nl_generation/      # NL查询生成（反向生成）
│   ├── dataset_construction/ # 数据集构建
│   └── data/               # 数据文件
├── taco_sql/               # TACO-SQL流程
│   ├── question_rewriting/  # 问题转写
│   ├── table_linking/      # 表格检索（Table Linking）
│   ├── query_planning/     # 查询规划与拆解
│   └── sql_generation/     # SQL生成
├── experiments/             # 实验框架
│   ├── baselines/          # 基线模型实验
│   ├── taco_sql_exp/       # TACO-SQL消融实验
│   └── evaluation/         # 评估工具
├── models/                 # 模型文件
├── configs/                # 配置文件
├── utils/                  # 工具函数
└── docs/                   # 文档
```

## 快速开始

### 1. 环境配置

```bash
# 创建conda环境
conda create -n taco python=3.8
conda activate taco

# 安装依赖
pip install -r requirements.txt
```

### 2. Benchmark构建

```bash
cd benchmark
# 按照README.md中的步骤构建benchmark
```

### 3. TACO-SQL使用

```bash
cd taco_sql
# 训练Table Linking模型
cd table_linking/training
python finetune_sbert.py

# 使用完整Pipeline
python pipeline/taco_sql_pipeline.py
```

### 4. 运行实验

```bash
cd experiments
# 运行基线实验
cd baselines/base_llm/GPT-o1
python run_experiment.py

# 运行消融实验
cd ../taco_sql_exp
python run_ablation.py
```

## 文档

- [Benchmark构建文档](benchmark/README.md)
- [TACO-SQL框架文档](taco_sql/README.md)
- [实验框架文档](experiments/README.md)
- [项目架构详解](old/docs/TACO项目架构与实验流程详解.md)
- [代码模块说明](old/docs/代码模块详细说明.md)

## 数据

- **TACO-SmartCity**: 1,500个真实查询（来自北京智慧城市数据服务）
- **TACO-OpenData**: 13,000个合成查询（来自北京和美国开放数据门户）

## 引用

如果使用本项目，请引用：

```bibtex
@article{taco2025,
  title={TACO: A Benchmark for Open-Domain Text-to-SQL with Ambiguous and Cross-Database Queries},
  author={...},
  journal={PVLDB},
  year={2025}
}
```

## 许可证

本项目遵循相应的开源许可证。

## 联系方式

如有问题，请通过GitHub Issues联系。

