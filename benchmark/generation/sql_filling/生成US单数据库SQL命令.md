# US数据集单数据库SQL生成命令

## 目标分配

总目标：**6,000条SQL**

| 类型 | 占比 | 目标数量 | 当前数量 | 还需生成 |
|------|------|---------|---------|---------|
| 单数据库 | 80.5% | **4,830条** | 2,513条 | **2,317条** |
| 跨2个数据库 | 15.0% | **900条** | 0条 | **900条** |
| 跨3个数据库 | 4.4% | **264条** | 0条 | **264条** |
| 跨4个数据库 | 0.1% | **6条** | 0条 | **6条** |
| **总计** | 100% | **6,000条** | 2,513条 | **3,487条** |

## 单数据库SQL生成

### 各数据库目标数量（平均220条/数据库）

| 数据库名称 | 当前数量 | 目标数量 | 还需生成 |
|-----------|---------|---------|---------|
| City of New York - 2516 | 68 | 220 | 152 |
| Department of Agriculture - 500 | 70 | 220 | 150 |
| Department of... - 247 | 77 | 220 | 143 |
| Louisville Metro Government - 446 | 78 | 220 | 142 |
| City of New Orleans - 171 | 82 | 220 | 138 |
| Montgomery County... - 454 | 96 | 220 | 124 |
| National Oceanic... - 520 | 96 | 220 | 124 |
| Cook County of Illinois - 433 | 101 | 220 | 119 |
| City of Austin - 1586 | 103 | 220 | 117 |
| City of Los Angeles - 352 | 105 | 220 | 115 |
| Vermont Center for... - 96 | 110 | 220 | 110 |
| Department of the Interior - 1006 | 119 | 220 | 101 |
| State of Washington - 604 | 119 | 220 | 101 |
| U.S. Department of... - 1451 | 124 | 220 | 96 |
| City of Chicago - 854 | 126 | 220 | 94 |
| National Institute... - 151 | 130 | 220 | 90 |
| State of Iowa - 362 | 133 | 220 | 87 |
| State of Connecticut - 845 | 143 | 220 | 77 |
| State of Maryland - 502 | 145 | 220 | 75 |
| State of Oregon - 596 | 155 | 220 | 65 |
| City of Seattle - 171 | 161 | 220 | 59 |
| State of Missouri - 145 | 172 | 220 | 48 |

## 生成命令

### 单个数据库生成

```bash
cd /home/u2023103807/TACO/benchmark/generation/sql_filling

python3 generate_us_single_db_sqls.py \
    --database_name "City of New York - 2516" \
    --target_count 220 \
    --max_retries 3
```

### 批量生成（按优先级）

#### 优先级1：需要生成最多的数据库（150+条）

```bash
# City of New York - 2516 (152条)
python3 generate_us_single_db_sqls.py --database_name "City of New York - 2516" --target_count 220

# Department of Agriculture - 500 (150条)
python3 generate_us_single_db_sqls.py --database_name "Department of Agriculture - 500" --target_count 220

# Department of... - 247 (143条)
python3 generate_us_single_db_sqls.py --database_name "Department of... - 247" --target_count 220

# Louisville Metro Government - 446 (142条)
python3 generate_us_single_db_sqls.py --database_name "Louisville Metro Government - 446" --target_count 220

# City of New Orleans - 171 (138条)
python3 generate_us_single_db_sqls.py --database_name "City of New Orleans - 171" --target_count 220
```

#### 优先级2：需要生成100-150条的数据库

```bash
# Montgomery County... - 454 (124条)
python3 generate_us_single_db_sqls.py --database_name "Montgomery County... - 454" --target_count 220

# National Oceanic... - 520 (124条)
python3 generate_us_single_db_sqls.py --database_name "National Oceanic... - 520" --target_count 220

# Cook County of Illinois - 433 (119条)
python3 generate_us_single_db_sqls.py --database_name "Cook County of Illinois - 433" --target_count 220

# City of Austin - 1586 (117条)
python3 generate_us_single_db_sqls.py --database_name "City of Austin - 1586" --target_count 220

# City of Los Angeles - 352 (115条)
python3 generate_us_single_db_sqls.py --database_name "City of Los Angeles - 352" --target_count 220

# Vermont Center for... - 96 (110条)
python3 generate_us_single_db_sqls.py --database_name "Vermont Center for... - 96" --target_count 220
```

#### 优先级3：需要生成50-100条的数据库

```bash
# Department of the Interior - 1006 (101条)
python3 generate_us_single_db_sqls.py --database_name "Department of the Interior - 1006" --target_count 220

# State of Washington - 604 (101条)
python3 generate_us_single_db_sqls.py --database_name "State of Washington - 604" --target_count 220

# U.S. Department of... - 1451 (96条)
python3 generate_us_single_db_sqls.py --database_name "U.S. Department of... - 1451" --target_count 220

# City of Chicago - 854 (94条)
python3 generate_us_single_db_sqls.py --database_name "City of Chicago - 854" --target_count 220

# National Institute... - 151 (90条)
python3 generate_us_single_db_sqls.py --database_name "National Institute... - 151" --target_count 220

# State of Iowa - 362 (87条)
python3 generate_us_single_db_sqls.py --database_name "State of Iowa - 362" --target_count 220

# State of Connecticut - 845 (77条)
python3 generate_us_single_db_sqls.py --database_name "State of Connecticut - 845" --target_count 220

# State of Maryland - 502 (75条)
python3 generate_us_single_db_sqls.py --database_name "State of Maryland - 502" --target_count 220
```

#### 优先级4：需要生成少于50条的数据库

```bash
# State of Oregon - 596 (65条)
python3 generate_us_single_db_sqls.py --database_name "State of Oregon - 596" --target_count 220

# City of Seattle - 171 (59条)
python3 generate_us_single_db_sqls.py --database_name "City of Seattle - 171" --target_count 220

# State of Missouri - 145 (48条)
python3 generate_us_single_db_sqls.py --database_name "State of Missouri - 145" --target_count 220
```

## 参数说明

- `--database_name`: 数据库名称（必须，注意引号）
- `--target_count`: 目标生成数量（可选，默认生成所有skeleton）
- `--max_retries`: 最大重试次数（默认3）
- `--config`: 配置文件路径（默认：./config.yaml）

## 注意事项

1. **数据库名称必须用引号包裹**，因为包含空格和特殊字符
2. **脚本会自动检查已有SQL数量**，只生成不足的部分
3. **所有生成的SQL都是英文的**，prompt已修改为英文版本
4. **建议按优先级顺序生成**，先处理需要生成最多的数据库

## 检查生成进度

```bash
# 统计当前各数据库的SQL数量
cd /home/u2023103807/TACO
for db in benchmark/data/us/output/single/*/; do 
    echo "$(basename "$db"): $(ls "$db"/*.json 2>/dev/null | wc -l)"; 
done | sort -t: -k2 -n
```

