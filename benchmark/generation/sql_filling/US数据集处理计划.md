# US数据集处理计划

## 当前状态

### US数据集结构
- **位置**: `old/saturn/TACO-Benchmark/us/data/`
- **数据库数量**: 1个（Vermont Center for... - 96）
- **数据库格式**: JSON格式，顶层键是表名（非标准schema格式）
- **SQL骨架**: 20个（在`new_sql_skeletons.json`中，格式为列表）

### 需要完成的步骤

1. **创建US数据集的database_chinese目录结构**
   - 将US数据集的数据库转换为标准格式
   - 创建`benchmark/data/us/database_chinese/`目录

2. **提取Schema**
   - 使用`extract_schema_chinese.py`提取标准schema格式

3. **生成SQL骨架文件**
   - 从`new_sql_skeletons.json`中提取SQL骨架
   - 按数据库分组，生成`*_sql_skeleton.json`文件

4. **生成图文件**
   - 使用`1build_schema_graphs_improved.py`生成图文件

5. **运行SQL填充**
   - 使用`2fill_sql_placeholders_improved.py`填充SQL骨架

## 处理步骤

### 步骤1: 准备US数据集目录结构

```bash
# 创建US数据集目录结构
mkdir -p benchmark/data/us/database_chinese
mkdir -p benchmark/data/us/output/sql_skeleton
mkdir -p benchmark/data/us/output/graph_chinese
mkdir -p benchmark/data/us/output/single
```

### 步骤2: 转换数据库格式

需要将US数据集的数据库JSON格式转换为标准格式（类似beijing的database_chinese）。

### 步骤3: 提取SQL骨架

从`new_sql_skeletons.json`中提取SQL骨架，按数据库分组。

### 步骤4: 生成图文件

使用改进的图生成脚本。

### 步骤5: 运行SQL填充

使用改进的SQL填充脚本。

## 注意事项

1. **US数据集使用英文表名**：不需要中文表名处理，但需要确保表名和列名正确引用
2. **SQL骨架格式**：需要从列表格式转换为按数据库分组的格式
3. **数据库数量少**：US数据集只有1个数据库，处理相对简单

