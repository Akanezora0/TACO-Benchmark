# US数据集重新生成计划

## 一、已完成的工作

### 1. ✅ 删除0表数据库
- **State of Hawaii - 136**: 已删除（0个表）
- **State of Oklahoma - 330**: 已删除（0个表）
- **剩余数据库数量**: 22个

### 2. ✅ 创建重新生成脚本

#### SQL骨架生成（500条）
- **脚本**: `benchmark/generation/sql_skeleton_generation/regenerate_us_skeletons_500.sh`
- **功能**: 为US数据集重新生成500个SQL骨架（之前是200个）
- **参数**: `--total_skeletons 500`

#### 图生成（500个）
- **脚本**: `benchmark/generation/sql_filling/regenerate_us_graphs_500.sh`
- **功能**: 基于500个SQL骨架生成500个图
- **说明**: 图生成脚本会自动处理所有SQL骨架，所以如果SQL骨架有500个，图也会生成500个

#### SQL填充（目标200条）
- **脚本**: `benchmark/generation/sql_filling/regenerate_us_sql_500.sh`
- **功能**: 基于500个SQL骨架填充SQL，目标是每个数据库200条完整SQL
- **说明**: 由于成功率约50%，500个骨架应该能生成约200条完整SQL

#### 完整流程脚本
- **脚本**: `benchmark/generation/sql_filling/regenerate_us_all_500.sh`
- **功能**: 依次执行上述三个步骤，完成整个重新生成流程

#### 结果检查脚本
- **脚本**: `benchmark/generation/sql_filling/check_us_sql_count.sh`
- **功能**: 检查每个数据库的SQL生成数量，统计完成情况

## 二、执行计划

### 方式1：分步执行（推荐，便于监控）

```bash
# 步骤1: 生成500个SQL骨架
cd /home/u2023103807/TACO
./benchmark/generation/sql_skeleton_generation/regenerate_us_skeletons_500.sh

# 步骤2: 生成500个图
./benchmark/generation/sql_filling/regenerate_us_graphs_500.sh

# 步骤3: 填充SQL（目标200条）
./benchmark/generation/sql_filling/regenerate_us_sql_500.sh

# 步骤4: 检查结果
./benchmark/generation/sql_filling/check_us_sql_count.sh
```

### 方式2：一键执行（自动完成所有步骤）

```bash
cd /home/u2023103807/TACO
./benchmark/generation/sql_filling/regenerate_us_all_500.sh
```

## 三、预期结果

### 目标
- **每个数据库**: 200条完整SQL
- **总数据库数**: 22个（已删除2个0表数据库）
- **总SQL数量**: 约4,400条（22 × 200）

### 成功率估算
- **当前平均成功率**: 约50%
- **SQL骨架数量**: 500个
- **预期生成SQL**: 500 × 50% = 250条
- **目标**: 200条（保守估计，实际可能更多）

## 四、注意事项

1. **执行时间**: 
   - SQL骨架生成：可能需要较长时间（22个数据库）
   - 图生成：相对较快
   - SQL填充：最耗时（需要调用LLM API）

2. **API费用**: 
   - SQL填充阶段会调用大量LLM API
   - 建议监控API使用情况

3. **错误处理**: 
   - 如果某个步骤失败，可以单独重新运行该步骤
   - 脚本会跳过已存在的文件，支持断点续传

4. **监控进度**: 
   - 可以查看各步骤的输出日志
   - 使用`check_us_sql_count.sh`随时检查进度

## 五、后续步骤

完成SQL填充后：
1. ✅ 检查每个数据库是否达到200条SQL
2. ⏳ 如果某些数据库不足200条，可以：
   - 分析失败原因
   - 调整参数（如增加重试次数）
   - 或接受当前结果（如果接近200条）
3. ⏳ 重新生成NL查询（基于新的SQL）

## 六、文件清单

### 已创建的脚本
- `benchmark/generation/sql_skeleton_generation/regenerate_us_skeletons_500.sh`
- `benchmark/generation/sql_filling/regenerate_us_graphs_500.sh`
- `benchmark/generation/sql_filling/regenerate_us_sql_500.sh`
- `benchmark/generation/sql_filling/regenerate_us_all_500.sh`
- `benchmark/generation/sql_filling/check_us_sql_count.sh`

### 相关文档
- `benchmark/generation/sql_filling/SQL填充失败原因分析.md`
- `benchmark/generation/sql_filling/US数据集重新生成计划.md`（本文档）

