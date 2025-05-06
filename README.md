# TACO-Benchmark

TACO-Benchmark is a Text-to-SQL benchmark in open-domain scenarios. The project includes three main datasets: us, beijing, and smartcity. The us and beijing datasets are synthesized by our data pipeline from open data portals and are freely available for [download](https://drive.google.com/file/d/1TcQOZoU2SkQQA5COQIux2cTIuMY1uzCi/view). The smartcity dataset is internal data and not fully open to the public, but you can contact us for access.

Note: Some code dependencies involve data privacy and may not be fully executable.

## Project Structure

### File Structure
```
TACO-Benchmark/
├── TACO-SQL/                # TACO-SQL Framework
│   ├── TACO-SQL.py         # Main framework implementation
│   ├── config.yaml         # Configuration file
│   ├── outputs/            # Generated SQL outputs
│   └── logs/               # Execution logs
├── us/                     # US Dataset
│   ├── code/              # Code Directory
│   │   ├── data_prep/     # Data Preprocessing Code
│   │   ├── nl_generation/ # Natural Language Query Generation Code
│   │   └── sql_generation/# SQL Generation Code
│   └── data/              # Data Directory
├── beijing/               # Beijing Dataset
│   ├── code/              # Code Directory
│   │   ├── data_prep/     # Data Preprocessing Code
│   │   ├── nl_generation/ # Natural Language Query Generation Code
│   │   └── sql_generation/# SQL Generation Code
│   └── data/              # Data Directory
└── smartcity/             # Smart City Dataset
```

## TACO-SQL Framework

TACO-SQL is a comprehensive framework for Text-to-SQL generation with multiple stages and approaches. It provides a unified interface for different SQL generation methods and includes various optimization techniques.

### Key Features

1. **Multiple Generation Methods**:
   - Base LLM Methods (GPT-4, DeepSeek, etc.)
   - LLM-Based Methods (DIN-SQL, MAC-SQL)
   - SFT-Based Methods (CodeS-33B, Qwen2.5-Coder)
   - Hybrid Methods (CHESS, Zero-NL2SQL)

2. **Pipeline Stages**:
   - Question Rewrite: Clarify and normalize user queries
   - Table Linking: Identify relevant database tables
   - Query Planning: Generate execution plans
   - SQL Generation: Convert plans to SQL using various methods

3. **Optimization Features**:
   - SQL Validation and Error Correction
   - JOIN Order Optimization
   - Query Structure Optimization
   - Multi-model Ensemble

### Usage

1. **Basic Usage**:
```python
from TACO_SQL import TACO_SQL_Pipeline, PipelineConfig, SQLGeneratorConfig, SQLGenerationType

# Create configuration
config = PipelineConfig(
    query_model_path="./retrieval/query_model",
    table_model_path="./retrieval/table_model",
    merged_table_csv="merged_table.csv",
    schema_json="schema.json",
    top_k_tables=5,
    sql_generator_config=SQLGeneratorConfig(
        generation_type=SQLGenerationType.HYBRID,
        model_name="deepseek-chat",
        api_key="your_api_key",
        base_url="https://api.deepseek.com"
    )
)

# Initialize pipeline
pipeline = TACO_SQL_Pipeline(config)

# Generate SQL
query = "Find all employees hired after 2020 with their department names"
sql = pipeline.run(query)
print(sql)
```

2. **Command Line Usage**:
```bash
python TACO-SQL.py --config config.yaml --query "your query here" --model GPT-o1
```

3. **Configuration File**:
```yaml
query_model_path: "./retrieval/query_model"
table_model_path: "./retrieval/table_model"
merged_table_csv: "merged_table.csv"
schema_json: "schema.json"
top_k_tables: 5
api_key: "your_api_key"
base_url: "https://api.deepseek.com"
model_path: ""
device: "cuda:0"
temperature: 0.1
top_p: 0.9
data_path: "path/to/data"
tables_json: "path/to/tables.json"
dataset: "beijing"
```

### Code Module Description

#### 1. Data Preprocessing Module (data_prep/)

The data preprocessing module is responsible for data cleaning, transformation, and preparation. Main functions include:

- **Data Cleaning**: Handling outliers and missing values in raw data
- **Data Transformation**: Converting data to standard format
- **Data Validation**: Ensuring data quality and consistency

#### 2. SQL Generation Module (sql_generation/)

The SQL generation module is responsible for generating SQL query skeletons and structures. Main functions include:

- **SQL Framework Extraction**: Extracting SQL query frameworks from raw data
- **AST and CFG Rule Generation**: Generating abstract syntax tree and control flow graph rules
- **SQL Skeleton Generation**: Generating SQL skeleton query structures based on rules
- **SQL Skeleton Filling and Validation**: Filling SQL skeletons and generating and validating complete SQL queries

#### 3. Natural Language Generation Module (nl_generation/)

The natural language generation module is responsible for converting SQL queries to natural language queries. Main functions include:

- **Database Schema Loading**: Loading and parsing database schemas
- **SQL Schema Graph Construction**: Building schema graphs for SQL queries
- **SQL Framework Parsing**: Parsing SQL query frameworks
- **Natural Language Query Generation**: Generating corresponding natural language queries

## Running Requirements

### Environment Requirements
- Python 3.7+

### Dependencies
```bash
pip install sqlparse sqlglot networkx tqdm torch openai transformers sentence-transformers nltk pandas pyyaml
```

## Data Preparation

### Configuration File Preparation
- Provide correct API keys and model names
- Set correct data paths
- Ensure database connection configuration is correct

---

## Data Examples

### 1. US Dataset

#### Single Database Query Example
```sql
SELECT landfills."LATITUDE", landfills."LONGITUDE" 
FROM landfills 
INNER JOIN outdoor_recreation_sites_inventory 
ON landfills."OBJECTID" = outdoor_recreation_sites_inventory."OBJECTID";
```

**Natural Language Query**:
"As an environmental scientist, I am evaluating the impact of new outdoor recreation sites on existing landfills. I need to collect the latitude and longitude coordinates of landfills and outdoor recreation areas to understand their spatial relationships, which is crucial for assessing environmental impacts, safety concerns, and future development planning."

**Query Intent**:
This query aims to retrieve latitude and longitude data from two tables (landfills and outdoor recreation sites inventory). By joining these tables based on the OBJECTID field, it analyzes the spatial relationships between landfills and outdoor recreation sites.

#### Cross-database Query Example
```sql
SELECT v."LATITUDE", v."LONGITUDE", d."DEPARTMENT_NAME"
FROM "Vermont Center for Geographic Information"."landfills" v
INNER JOIN "U.S. Department of Agriculture"."recreation_sites" d
ON v."OBJECTID" = d."SITE_ID";
```

**Natural Language Query**:
"I am currently conducting a spatial analysis study, specifically aiming to analyze the spatial relationships between various recreation sites managed by the U.S. Department of Agriculture (USDA), such as national forests, recreation parks, or other outdoor activity areas, and landfills recorded by the Vermont Center for Geographic Information (VCGI). To complete this analysis, I need to obtain precise latitude and longitude coordinate data for these two types of locations and understand which government agencies or management departments they belong to."

**Query Intent**:
This query aims to analyze the spatial relationships between recreation sites and landfills managed by different government departments, obtaining their latitude and longitude coordinates and department affiliation information.

### 2. Beijing Dataset

#### Single Database Query Example
```sql
SELECT "shiminzuzongjiaoweiguifanhuaqingzhenshipinzhuanguishuju3076.专用标识牌号", 
       "shiminzuzongjiaoweiguifanhuaqingzhenshipinzhuanguishuju3076.专柜名称", 
       "shiminzuzongjiaoweiguifanhuaqingzhenshipinzhuanguishuju3076.地址",
       "shishichangjianguanjucanyinfuwuxukezhengqingkuangxinxi3374.JYZMC",
       "shishichangjianguanjucanyinfuwuxukezhengqingkuangxinxi3374.XKZBH",
       "shishichangjianguanjucanyinfuwuxukezhengqingkuangxinxi3374.JYCS",
       "shishichangjianguanjucanyinfuwuxukezhengqingkuangxinxi3374.JYXM",
       "shishichangjianguanjucanyinfuwuxukezhengqingkuangxinxi3374.FZRQ",
       "shishichangjianguanjucanyinfuwuxukezhengqingkuangxinxi3374.YXQZ"
FROM "shiminzuzongjiaoweiguifanhuaqingzhenshipinzhuanguishuju3076"
INNER JOIN "shishichangjianguanjucanyinfuwuxukezhengqingkuangxinxi3374"
ON "shiminzuzongjiaoweiguifanhuaqingzhenshipinzhuanguishuju3076.专用标识牌号" = "shishichangjianguanjucanyinfuwuxukezhengqingkuangxinxi3374.XKZBH";
```

**Natural Language Query**:
"市民反映，在社区附近的清真食品专柜买东西时，想确认一下这个专柜是不是正规合法的。专柜上挂着一个牌子，上面写着'专用标识牌号：QZ-2023-018'，但是没看到餐饮服务许可证。听说有些专柜可能没有许可证还在经营，担心买到不合规的清真食品。能不能帮忙查一下这个专柜有没有对应的餐饮服务许可证？包括经营者的名字、经营项目、许可证的有效期这些信息。"

**Query Intent**:
该查询旨在验证清真食品专柜的合法性，通过连接清真食品专柜归属数据表和餐饮服务许可证信息表，获取专柜的基本信息和对应的许可证详细信息，确保专柜经营合法合规。

#### Cross-database Query Example
```sql
SELECT a."检查日期", a."检查结果", c."专柜名称", c."地址"
FROM "安全生活"."食品安全检查记录" a
INNER JOIN "餐饮美食"."清真食品专柜归属数据" c
ON a."专柜编号" = c."专用标识牌号"
WHERE a."检查结果" = '不合格';
```

**Natural Language Query**:
"最近部门在检查清真食品专柜的食品安全问题，我想查询最近被检查出食品安全问题的清真食品专柜信息，包括检查日期、具体问题、专柜名称和地址，以便了解这些不合格专柜的分布情况。"

**Query Intent**:
该查询旨在获取食品安全检查不合格的清真食品专柜信息，包括检查时间、问题详情和位置信息，用于分析不合格专柜的分布情况。

### 3. Smart City Dataset

#### Single Database Query Example
```sql
SELECT yh_ms as 隐患名称 
FROM gjk.o_ql_yjglj_csaqyhtz 
WHERE jc_dz LIKE '%xx家园A区%';
```

**Natural Language Query**:
"xx家园 A 区存在重大安全隐患，请求相关部门落实整改。xx家园 A 区小区内部存在易燃垃圾长期无人清理、废弃电动车自行车随意堆放的重大火灾安全隐患，图片展示的仅为小区内部北侧和东侧栅栏处，长期存在易燃垃圾无人清理，电动车充电不规范，若发生爆炸火灾后果不堪设想，小区居民人身财产安全无法保证。"

**Query Intent**:
该查询旨在获取xx家园A区存在的安全隐患信息，通过地址匹配查询安全隐患表中的相关记录，了解具体的安全隐患情况。

#### Cross-database Query Example
```sql
SELECT valid 
FROM tccsj_bw 
WHERE parking_id IN (
    SELECT position_id 
    FROM tccsj 
    WHERE parking_name LIKE '%黄石小区%'
);
```

**Natural Language Query**:
"市民反映，黄石小区停车场有几辆汽车停了好几个月没有动，停车场的停车位很少，影响自己停车，具体地址：xx区xx街道东黄石小区停车场，没有和物业沟通过，来电希望解决停车场车不动问题。"

**Query Intent**:
该查询旨在检查黄石小区停车场中长时间未移动的车辆信息，通过连接停车场位置表和车辆信息表，获取这些车辆的停放状态，用于解决停车位占用问题。

