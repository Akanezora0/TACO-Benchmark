# TACO: A Benchmark for Open-Domain Text-to-SQL

**TACO (Text-to-SQL with Ambiguous and Cross-Database Queries)** is a comprehensive benchmark designed to evaluate Text-to-SQL systems in real-world open-domain scenarios. Unlike traditional closed-domain benchmarks (e.g., Spider, BIRD) that focus on single predefined databases, TACO addresses the fundamental challenges encountered when users interact with large-scale, heterogeneous data lakes.

## 🎯 Benchmark Overview

TACO consists of **14,500 high-quality Text-to-SQL examples** covering diverse domains including finance, healthcare, transportation, housing, and government services. The benchmark is designed to evaluate how well Text-to-SQL systems handle three critical challenges that are common in real-world applications but underrepresented in existing benchmarks:

1. **Ambiguous Natural Language Questions** - Users often have limited knowledge of the underlying data and issue vague or redundant questions with unclear intent or incomplete constraints
2. **Unspecified Target Databases** - Questions rarely identify which database or table is relevant, requiring systems to retrieve candidate tables from large, heterogeneous data lakes
3. **Cross-Database Querying** - Answering a single question may require combining data from multiple databases with weak or implicit relationships

## 📊 Dataset Statistics

### TACO-Beijing Dataset

- **Single-Database Examples**: 
    - SQL queries: 4,028
    - Natural language queries: 5,587
    - Databases: 24
    - Domains: Finance & Taxation, Healthcare, Transportation, Housing, etc.

- **Cross-Database Examples**:
    - SQL queries: 466
    - Databases involved: 2-4 databases per query
    - Query types: JOIN and UNION operations

### TACO-US Dataset

- **Single-Database Examples**:
    - Natural language queries: 3,990
    - Databases: 22
    - Domains: City services, Agriculture, Interior, Healthcare, etc.

### Data Distribution

- **Single-database queries**: ~80.5%
- **2-database queries**: ~15.0%
- **3-database queries**: ~4.4%
- **4-database queries**: ~0.1%

## 🔍 Core Challenges and Examples

### Challenge 1: Ambiguous Natural Language Questions

Users often have limited knowledge of the underlying data and issue vague or redundant questions with unclear intent or incomplete constraints. These queries contain:

- **Redundant information**: Extra context that doesn't directly map to SQL
- **Implicit constraints**: Requirements that are implied but not explicitly stated
- **Ambiguous expressions**: Vague terms that require interpretation

#### Example 1.1: Redundant Context

**Natural Language Query**:

```
I need to verify the government fund budget revenue for the entire year of 2018.
Please help me summarize the budget revenue and actual received amounts for all 
government fund budget projects in 2018, organized by different budget projects.
I want to see the differences between budgeted and actual revenue, find projects with 
large deviations, and confirm whether budget execution is reasonable. This data is 
very important for annual settlement review and reporting to the Finance Department.
I hope to have a detailed and clearly categorized revenue execution report.
```

**Key Ambiguities**:

- Multiple overlapping requirements (summarize, categorize and organize, find differences)
- Vague terms ("large revenue deviation" - what threshold?)
- Redundant explanation of use case

**SQL Query**:

```sql
SELECT 
  "finance_bureau_budget_execution_report"."ProjectName",
  "finance_bureau_budget_execution_report"."BudgetRevenue2018",
  "finance_bureau_budget_execution_report"."Year"
FROM "finance_bureau_budget_execution_report"
WHERE "finance_bureau_budget_execution_report"."Year" = '2018'
```

#### Example 1.2: Implicit Constraints

**Natural Language Query**:

```
I need to look at birth rate trends for community area 1, especially focusing on changes 
between 2022 and 2023. This will help understand population growth patterns 
and maternal health status in that area, which is important for evaluating the 
effectiveness of health intervention programs.
```

**Key Ambiguities**:

- "trends" implies time series, but query only specifies one area
- "especially focusing on 2022 and 2023" but mentions "from 2021 to 2023"
- "health intervention programs" - which specific programs?
- Implicit need for comparison across years

**SQL Query**:

```sql
SELECT 
  "public_health_statistics__selected_public_health_indicators_by_chicago_community_area__historical"."Community Area",
  "public_health_statistics__selected_public_health_indicators_by_chicago_community_area__historical"."Birth Rate"
FROM "public_health_statistics__selected_public_health_indicators_by_chicago_community_area__historical"
WHERE "public_health_statistics__selected_public_health_indicators_by_chicago_community_area__historical"."Community Area" = '1'
```

#### Example 1.3: Vague Intent

**Natural Language Query**:

```
I need to query the revenue and expenditure status of all government funds for 2020,
with special focus on the revenue sources and expenditure details of each fund.
I hope to obtain information on fund usage for infrastructure construction and social 
welfare projects, as well as whether there are over-budget expenditures or non-compliant 
fund usage.
```

**Key Ambiguities**:

- "all government funds" - all funds or specific categories?
- "especially focus on" - how to prioritize or filter?
- "over-budget expenditure" - what threshold defines "over-budget"?
- "non-compliant fund usage" - what defines compliance?

**SQL Query**:

```sql
SELECT 
  "government_fund_budget_report"."ExchangeTime",
  "government_fund_budget_report"."RevenueItem",
  "government_fund_budget_report"."ExpenditureAmount"
FROM "government_fund_budget_report"
WHERE "government_fund_budget_report"."Year" = '2020'
```

#### Example 1.4: Redundant Context with Multiple Overlapping Requirements

**Natural Language Query**:

```
I need comprehensive information about all critical wildfire crisis strategy 
landscapes in California. Specifically, I want to see the landscape names, exact 
geographic locations, acreage, current risk levels, and coverage of major fire 
prevention measures. Please filter for high-risk areas and prioritize landscapes 
that have historically experienced major fires or are designated as fire prevention 
priorities. This data needs to be clean and exportable for reports and presentations 
to help efficiently evaluate and manage fire prevention resources.
```

**Key Ambiguities**:

- "comprehensive information" - which specific fields are needed?
- "high-risk areas" - what defines high risk? (no threshold specified)
- "prioritize landscapes" - how to rank or order them?
- "historically experienced major fires" - what time period? What defines "major"?
- Redundant explanation of use case and deadline information

**SQL Query**:

```sql
SELECT 
  "wildfire_crisis_strategy_landscapes_feature_layer"."OBJECTID",
  "wildfire_crisis_strategy_landscapes_feature_layer"."NAME",
  "wildfire_crisis_strategy_landscapes_feature_layer"."STATE"
FROM "wildfire_crisis_strategy_landscapes_feature_layer"
WHERE "wildfire_crisis_strategy_landscapes_feature_layer"."STATE" = 'California'
ORDER BY "wildfire_crisis_strategy_landscapes_feature_layer"."OBJECTID"
LIMIT 10
```

#### Example 1.5: Implicit Constraints and Vague Aggregation Requirements

**Natural Language Query**:

```
I need to analyze student participation rates in career and technical education 
programs across different school districts. Can you show me the number of students 
who meet the participation criteria and the total student population for each 
district? I want to calculate participation and completion rates to evaluate 
program effectiveness, identify high-performing and underperforming districts, 
and use this data to inform resource allocation and policy adjustments. The data 
should be organized by district for easy analysis.
```

**Key Ambiguities**:

- "participation criteria" - what specific criteria? (not explicitly stated)
- "high-performing and underperforming" - what thresholds define performance?
- "calculate participation and completion rates" - but query only requests counts, not rates
- "resource allocation" - which resources? (irrelevant to SQL query)
- Implicit need for aggregation that isn't explicitly requested

**SQL Query**:

```sql
SELECT 
  "cte_perkins_concentrators_2022_cohort"."Districtname",
  "cte_perkins_concentrators_2022_cohort"."NumberofStudentsInNumerator",
  "cte_perkins_concentrators_2022_cohort"."NumberofStudentsInDenominator"
FROM "cte_perkins_concentrators_2022_cohort"
WHERE "cte_perkins_concentrators_2022_cohort"."schoolyear" = '2022'
```

#### Example 1.6: Ambiguous Temporal and Comparative Requirements

**Natural Language Query**:

```
I'm interested in comparing housing permit data from recent years to understand 
growth patterns. Can you show me permit information for residential construction 
projects, focusing on the differences between single-family and multi-family units? 
I need this to identify areas with significant development activity and assess 
whether current zoning policies are effectively managing growth.
```

**Key Ambiguities**:

- "recent years" - which specific years? (no time range specified)
- "differences between single-family and multi-family" - what kind of comparison? (count, percentage, trend?)
- "significant development activity" - what threshold defines "significant"?
- "effectively managing growth" - what metrics define effectiveness?
- Redundant context about city council meeting (irrelevant to query)

### Challenge 2: Unspecified Target Databases

Questions rarely identify which database or table is relevant. Systems must retrieve candidate tables from large, heterogeneous data lakes without explicit guidance.

#### Example 2.1: No Database Mentioned

**Natural Language Query**:

```
I want to understand the traffic congestion situation in the area in recent years, 
especially data during morning and evening rush hours. I need to see the congestion 
levels in different regions and different road segments, and whether there is an 
improvement trend.
```

**Challenge**:

- No mention of which database contains traffic data
- Geographic area could be in multiple databases (Traffic Services, Municipal Traffic Commission, etc.)
- "traffic congestion" could map to various tables across different databases
- System must identify relevant databases and tables

**Possible Target Databases**:

- Traffic Services
- Municipal Traffic Commission

#### Example 2.2: Vague Domain Reference

**Natural Language Query**:

```
I'm researching public health trends in urban areas. Can you show me data about 
community health indicators, particularly birth rates and mortality statistics 
for different neighborhoods? I'm interested in understanding health disparities 
across the city.
```

**Challenge**:

- "urban areas" - which city? (Chicago, New York, Los Angeles?)
- "community health indicators" - could be in multiple health-related databases
- "different neighborhoods" - requires identifying correct geographic granularity
- System must search across multiple databases to find relevant tables

**Possible Target Databases**:

- City of Chicago - 854 (public health statistics)
- City of New York - 2516 (health department data)
- Various state health databases

#### Example 2.3: Implicit Domain Knowledge

**Natural Language Query**:

```
I need to review the budget execution status of government funds, focusing on 
projects that may have issues, such as abnormal expenditures or budget overruns.
```

**Challenge**:

- "government funds" - could be in Finance & Taxation, Municipal Finance Bureau, or other finance databases
- "budget execution status" - multiple tables might contain budget execution data
- "potentially problematic projects" - requires understanding what defines "problematic"
- System must identify the correct database and understand financial terminology

#### Example 2.4: Vague Geographic and Domain Reference

**Natural Language Query**:

```
I'm researching agricultural production trends and need data on crop yields, 
farm sizes, and production volumes. I'm particularly interested in understanding 
how different regions compare in terms of agricultural output and what factors 
might influence productivity. Can you help me find relevant datasets?
```

**Challenge**:

- "agricultural production" - could be in Department of Agriculture, state databases, or USDA databases
- "different regions" - which geographic level? (state, county, national?)
- "crop yields, farm sizes, production volumes" - multiple tables across different databases might contain this information
- No mention of specific database, table, or time period
- System must search across multiple agricultural databases to find relevant tables

**Possible Target Databases**:

- Department of Agriculture - 500 (federal agricultural data)
- State of Iowa - 362 (state agricultural statistics)
- Various state agricultural databases

#### Example 2.5: Implicit Domain Knowledge and Terminology

**Natural Language Query**:

```
I need to analyze educational outcomes for students in vocational training 
programs. Can you show me data on student enrollment, completion rates, and 
performance metrics? I want to understand which programs are most effective 
and where we might need to allocate additional resources.
```

**Challenge**:

- "vocational training programs" - could be CTE (Career and Technical Education) data in multiple state education databases
- "educational outcomes" - could refer to grades, test scores, graduation rates, or employment outcomes
- "performance metrics" - which specific metrics? (not specified)
- No mention of which state, school year, or database
- System must identify relevant education databases and understand educational terminology

**Possible Target Databases**:

- State of Washington - 604 (CTE Perkins concentrators data)
- State of Oregon - 596 (state education data)
- Various state education departments

#### Example 2.6: Ambiguous Resource and Location References

**Natural Language Query**:

```
I'm working on a conservation project and need information about protected 
areas, wildlife habitats, and land management practices. I want to understand 
the distribution of conservation efforts and identify areas that might need 
additional protection or management resources.
```

**Challenge**:

- "protected areas" - could be in Department of Interior, state parks databases, or environmental databases
- "wildlife habitats" - multiple databases might contain habitat information
- "land management practices" - could refer to forestry, agriculture, or conservation management
- "conservation efforts" - which type? (federal, state, local, private?)
- No geographic scope specified (national, state, or local?)
- System must search across multiple environmental and land management databases

**Possible Target Databases**:

- Department of the Interior - 1006 (federal land management)
- Department of Agriculture - 500 (forestry and conservation)
- State environmental databases

### Challenge 3: Cross-Database Querying

Answering a single question may require combining data from multiple databases with weak or implicit relationships, demanding multi-step query planning and result integration.

#### Example 3.1: Cross-Database JOIN

**Natural Language Query**:

```
Query the construction project completion filing information from the Housing database,
join with the public toilet information from the Life Services database,
count the number of public toilets for each project name,
and sort by the number of public toilets in descending order.
```

**Challenge**:

- Requires data from two different databases: Housing and Life Services
- Implicit relationship: matching by some identifier (sequence number)
- Need to understand table structure across databases
- Requires SQLite ATTACH DATABASE syntax

**SQL Query**:

```sql
SELECT 
  "Housing"."construction_project_completion_filing"."ProjectName",
  COUNT("LifeServices"."public_toilet_info"."SequenceNumber") AS "ToiletCount"
FROM "Housing"."construction_project_completion_filing"
JOIN "LifeServices"."public_toilet_info"
  ON "Housing"."construction_project_completion_filing"."SequenceNumber" 
   = "LifeServices"."public_toilet_info"."SequenceNumber"
GROUP BY "Housing"."construction_project_completion_filing"."ProjectName"
ORDER BY "ToiletCount" DESC
```

#### Example 3.2: Cross-Database UNION

**Natural Language Query**:

```
I want to view traffic violation records for all cities, including data from 
multiple cities, sorted by time to see the differences in violation patterns 
across different cities.
```

**Challenge**:

- Data from multiple city databases
- Need to combine similar but not identical table structures
- Requires UNION to merge results from different databases
- Time sorting across multiple data sources

**SQL Structure**:

```sql
SELECT * FROM "CityA"."traffic_violation_records"
UNION ALL
SELECT * FROM "CityB"."traffic_violation_records"
ORDER BY "Time"
```

#### Example 3.3: Multi-Database Aggregation

**Natural Language Query**:

```
I need to analyze healthcare spending across different states. Can you combine 
data from state health departments and federal health databases to show me 
total healthcare expenditures by state, including both public and private 
spending where available?
```

**Challenge**:

- Requires data from multiple state databases and federal databases
- Different schemas across databases
- Need to aggregate and integrate results
- Handle missing data across sources

**Possible Databases**:

- State of Maryland - 502 (state health data)
- State of Oregon - 596 (state health data)
- U.S. Department of... - 1451 (federal health data)

#### Example 3.4: Cross-Database JOIN with Implicit Relationship

**Natural Language Query**:

```
I'm conducting a study on the relationship between educational outcomes and 
public health indicators. Can you join student performance data from education 
databases with community health statistics to show me how health indicators 
correlate with academic achievement across different school districts?
```

**Challenge**:

- Requires data from education databases (e.g., State of Washington - 604) and health databases (e.g., City of Chicago - 854)
- Implicit relationship: matching by geographic area (school district vs. community area)
- Different geographic granularities may need to be reconciled
- Requires understanding of both educational and health terminology
- Need to handle schema differences and data type mismatches

**Possible Databases**:

- State of Washington - 604 (education data)
- City of Chicago - 854 (public health statistics)
- Other state education and health databases

#### Example 3.5: Cross-Database UNION with Schema Alignment

**Natural Language Query**:

```
I need to compile a comprehensive list of all professional licenses issued 
across multiple states. Can you combine license data from different state 
databases, showing me the license type, holder name, issue date, and status 
for all records? I want to see this data sorted chronologically to identify 
trends in professional licensing.
```

**Challenge**:

- Data from multiple state databases (e.g., State of Washington, State of Oregon, etc.)
- Similar but not identical table structures across states
- Column names and data formats may differ
- Requires UNION to merge results from different databases
- Need to align schemas and handle missing columns
- Chronological sorting across multiple data sources

**SQL Structure**:

```sql
SELECT "LicenseType", "Name", "IssueDate", "Status" 
FROM "State of Washington"."professional_licenses_issued_to_individuals_in_asotin_county"
UNION ALL
SELECT "LicenseType", "Name", "IssueDate", "Status" 
FROM "State of Oregon"."professional_licenses_table"
ORDER BY "IssueDate"
```

#### Example 3.6: Multi-Database Aggregation with Complex Relationships

**Natural Language Query**:

```
I'm analyzing federal land management and conservation efforts. I need to 
combine data from the Department of Interior on protected wilderness areas 
with Department of Agriculture data on wildfire management landscapes to 
create a comprehensive view of conservation priorities. Show me the overlap 
between these areas and calculate total protected acreage by state, including 
both wilderness areas and fire management zones.
```

**Challenge**:

- Requires data from Department of the Interior - 1006 (wilderness areas) and Department of Agriculture - 500 (wildfire landscapes)
- Different schemas and geographic representations
- Need spatial/geographic matching or relationship inference
- Complex aggregation across multiple databases
- Handle different area calculation methods
- Requires understanding of conservation and land management terminology

**Possible Databases**:

- Department of the Interior - 1006 (national wilderness areas)
- Department of Agriculture - 500 (wildfire crisis strategy landscapes)
- State databases (for state-level protected areas)

## 📁 Directory Structure

```
benchmark/
├── generation/                    # Data generation scripts
│   ├── sql_skeleton_generation/  # SQL skeleton generation
│   ├── sql_filling/              # SQL content filling
│   ├── nl_query/                 # Natural language query generation
│   └── cross_database/           # Cross-database SQL generation
├── data/                         # Generated benchmark data
│   ├── beijing/                  # Beijing dataset
│   │   ├── database_chinese/     # Database schemas (24 databases)
│   │   └── output/               # Generated data
│   │       ├── single/           # Single-database SQL queries
│   │       ├── nl_query/         # Natural language queries
│   │       ├── cross_db_final/   # Cross-database SQL queries
│   │       ├── sql_skeleton/     # SQL skeletons (intermediate)
│   │       ├── sql_structure/    # SQL structures (intermediate)
│   │       └── ast_cfg/          # AST/CFG files (intermediate)
│   └── us/                       # US dataset
│       ├── database/              # Database schemas (22 databases)
│       └── output/               # Generated data
│           ├── single/           # Single-database SQL queries
│           ├── nl_query/         # Natural language queries
│           └── ...               # Similar structure to beijing
└── README.md                     # This file
```

## 📄 Data Format

### Single-Database Example

Each single-database example contains:

```json
{
  "sql": "SELECT \"table_name\".\"column_name\" FROM \"table_name\" WHERE \"table_name\".\"column_name\" = 'value'",
  "sql_skeleton": "SELECT _ FROM _ WHERE _ = _",
  "natural_language_query": "Natural language description of the user query",
  "database": "Database name",
  "tables": {
    "table_name": ["column1", "column2", ...]
  },
  "metadata": {
    "has_join": false,
    "has_subquery": false,
    "has_aggregate": false
  },
  "cot_steps": {
    "step1_sql_analysis": "...",
    "step2_business_scenario": "...",
    "step3_user_scenario": "...",
    "step4_nl_generation": "..."
  }
}
```

### Cross-Database Example

Each cross-database example contains:

```json
{
  "sql": "SELECT ... FROM \"database1\".\"table_name\" JOIN \"database2\".\"table_name\" ON ...",
  "sql_skeleton": "SELECT _ FROM _ JOIN _ ON _",
  "databases": ["database1", "database2"],
  "table_database_mapping": {
    "table_name": "database_name"
  },
  "results": [[...], [...]],
  "metadata": {
    "num_databases": 2,
    "query_type": "JOIN"
  }
}
```

## 🚀 Usage

### Loading the Dataset

```python
import json
from pathlib import Path

# Load single-database examples
single_dir = Path("benchmark/data/beijing/output/single")
for db_dir in single_dir.iterdir():
    for sql_file in db_dir.glob("generated_sql_*.json"):
        with open(sql_file, 'r', encoding='utf-8') as f:
            example = json.load(f)
            sql = example['sql']
            nl_query = example.get('natural_language_query')
            database = example['database']
            # Process the example...

# Load cross-database examples
cross_db_dir = Path("benchmark/data/beijing/output/cross_db_final")
for sql_file in cross_db_dir.glob("cross_db_generated_sql_*.json"):
    with open(sql_file, 'r', encoding='utf-8') as f:
        example = json.load(f)
        sql = example['sql']
        databases = example['databases']
        results = example.get('results', [])
        # Process the example...
```

### Data Generation Pipeline

The benchmark data is generated through a three-step pipeline:

1. **SQL Skeleton Generation**: Generate SQL skeletons from CFG rules and expert examples
2. **SQL Content Filling**: Fill skeletons with concrete table names, column names, and values using LLMs
3. **NL Query Generation**: Generate natural language queries from SQL using Chain-of-Thought (CoT) approach

For detailed generation instructions, see the scripts in `benchmark/generation/`.

## 📚 Key Features

- **Real-World Complexity**: Queries reflect actual user behavior with ambiguity, redundancy, and implicit constraints
- **Open-Domain Setting**: Systems must identify relevant databases and tables from large data lakes
- **Cross-Database Support**: Includes queries spanning 2-4 databases with JOIN and UNION operations
- **Diverse Domains**: Covers finance, healthcare, transportation, housing, government services, and more
- **Multi-Domain Support**: Includes examples from multiple geographic regions and domains
- **Executable SQL**: All SQL queries are validated and executable against the provided databases

## 🔧 Requirements

- Python 3.8+
- Required packages: See `requirements.txt` in the main project directory
- LLM API access (for SQL filling and NL generation, if regenerating data)

## 📖 Related Work

This benchmark is part of the TACO project. For more information about:

- The TACO framework and methodology
- Baseline experiments and results
- Evaluation metrics

Please refer to the main project repository.
