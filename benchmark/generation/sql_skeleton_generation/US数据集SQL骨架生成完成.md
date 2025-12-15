# US数据集SQL骨架生成完成

## 完成的工作

### 1. 创建专家示例文件
- **文件**: `benchmark/data/target/expert_skeletons_us.json`
- **数量**: 41个专家示例
- **特点**: 
  - 涵盖多种SQL模式：SIMPLE_SELECT, WHERE, JOIN, SUBQUERY, AGGREGATE, UNION等
  - 难度分布：从简单查询到复杂JOIN和子查询
  - 多样性：包含各种SQL操作和模式

### 2. 生成SQL骨架
- **脚本**: `benchmark/generation/sql_skeleton_generation/generate_for_databases_improved.py`
- **处理数据库数**: 24个
- **每个数据库**: 200个SQL骨架
- **总计**: 4,800个SQL骨架

## 生成结果

所有24个US数据库都已成功生成SQL骨架：

1. State of Hawaii - 136: 200个
2. U.S. Department of... - 1451: 200个
3. National Institute... - 151: 200个
4. State of Maryland - 502: 200个
5. State of Oregon - 596: 200个
6. Department of the Interior - 1006: 200个
7. City of New Orleans - 171: 200个
8. City of Los Angeles - 352: 200个
9. Vermont Center for... - 96: 200个
10. Cook County of Illinois - 433: 200个
11. Louisville Metro Government - 446: 200个
12. State of Iowa - 362: 200个
13. Montgomery County... - 454: 200个
14. Department of... - 247: 200个
15. State of Washington - 604: 200个
16. City of New York - 2516: 200个
17. State of Oklahoma - 330: 200个
18. City of Chicago - 854: 200个
19. City of Austin - 1586: 200个
20. National Oceanic... - 520: 200个
21. City of Seattle - 171: 200个
22. State of Connecticut - 845: 200个
23. State of Missouri - 145: 200个
24. Department of Agriculture - 500: 200个

## 文件位置

- **SQL骨架文件**: `benchmark/data/us/output/sql_skeleton/`
- **AST/CFG文件**: `benchmark/data/us/output/ast_cfg/`
- **SQL结构文件**: `benchmark/data/us/output/sql_structure/`
- **专家示例文件**: `benchmark/data/target/expert_skeletons_us.json`

## 下一步

US数据集的SQL骨架已经生成完成，接下来可以：

1. **生成图文件**: 使用 `1build_schema_graphs_improved.py` 为US数据集生成图文件
2. **填充SQL骨架**: 使用 `2fill_sql_placeholders_improved.py` 填充SQL骨架，生成完整的SQL语句
3. **验证结果**: 检查生成的SQL骨架的质量和多样性

## 注意事项

1. **目录命名**: 当前US数据集仍使用`database_chinese`目录名，但实际是英文数据集。如果需要，可以重命名为`database`。
2. **专家示例**: `expert_skeletons_us.json`包含了41个专家示例，涵盖了各种SQL模式，确保生成的SQL骨架具有足够的多样性。
3. **SQL骨架格式**: 生成的SQL骨架是字符串数组格式，每个元素是一个SQL骨架模板（使用`_`作为占位符）。

