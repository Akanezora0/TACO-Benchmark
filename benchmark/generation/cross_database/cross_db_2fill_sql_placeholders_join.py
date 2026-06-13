#!/usr/bin/env python3
"""
Cross-database SQL skeleton filling script.

Extends the single-database SQL filling script to support cross-database scenarios:
1. Load schemas and graph files from multiple databases
2. Explicitly inform the LLM in the prompt that this is a cross-database query
3. Have the LLM generate SQL with database prefixes (e.g., database_name.table_name)
"""

import json
import os
import re
from tqdm import tqdm
import random
import sqlparse
import sqlite3
import networkx as nx
from openai import OpenAI
from collections import defaultdict
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import single-database functions
import sys
import importlib.util
sql_filling_dir = os.path.join(os.path.dirname(__file__), '..', 'sql_filling')
sys.path.insert(0, sql_filling_dir)

# Dynamic import
spec = importlib.util.spec_from_file_location(
    "fill_sql_placeholders_improved",
    os.path.join(sql_filling_dir, "fill_sql_placeholders.py")
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)

load_config = fill_module.load_config
get_client = fill_module.get_client
load_schema = fill_module.load_schema
analyze_sql_skeleton = fill_module.analyze_sql_skeleton
construct_enhanced_prompt = fill_module.construct_enhanced_prompt
# execute_sql may not exist; cross-database SQL does not require direct execution validation
load_graph_metadata = getattr(fill_module, 'load_graph_metadata', None)

def convert_to_single_database_sql(cross_db_sql, table_database_mapping):
    """Convert cross-database SQL to single-database SQL (remove database prefixes)."""
    single_db_sql = cross_db_sql
    # Replace "database_name"."table_name" with "table_name"
    for table, db in table_database_mapping.items():
        # Handle quoted identifiers
        pattern1 = rf'"{re.escape(db)}"\."{re.escape(table)}"'
        replacement1 = f'"{table}"'
        single_db_sql = re.sub(pattern1, replacement1, single_db_sql)

        # Handle unquoted identifiers
        pattern2 = rf'{re.escape(db)}\.{re.escape(table)}'
        replacement2 = table
        single_db_sql = re.sub(pattern2, replacement2, single_db_sql)

    # Replace "database_name"."table_name"."column_name" with "table_name"."column_name"
    for table, db in table_database_mapping.items():
        pattern = rf'"{re.escape(db)}"\."{re.escape(table)}"\."([^"]+)"'
        replacement = rf'"{table}"."\1"'
        single_db_sql = re.sub(pattern, replacement, single_db_sql)

    return single_db_sql

def execute_sql_on_database(sql, db_path):
    """Execute SQL on a single database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        # Empty results still count as success (consistent with single-database behavior)
        if results:
            return results, True
        else:
            return [], True  # Empty results still count as success
    except sqlite3.Error as e:
        return None, False
    except Exception as e:
        return None, False

def get_tables_in_database(db_path, alias=None):
    """Get all table names in a database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return set(tables)
    except:
        return set()

def validate_tables_in_sql(sql, databases, database_dir, table_database_mapping, db_aliases):
    """Verify that tables used in SQL exist in the corresponding databases."""
    # Extract table names used in SQL
    table_pattern = r'"(?:db\d+|[\u4e00-\u9fa5]+)"\."([^"]+)"'
    tables_in_sql = set(re.findall(table_pattern, sql))

    # Check whether each table exists in its database
    missing_tables = []
    for table_name in tables_in_sql:
        # Find the database for the table
        db_name = table_database_mapping.get(table_name)
        if db_name:
            db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
            if os.path.exists(db_path):
                tables_in_db = get_tables_in_database(db_path)
                if table_name not in tables_in_db:
                    missing_tables.append((table_name, db_name))

    return missing_tables

def execute_cross_database_sql_with_attach(cross_db_sql, databases, database_dir, table_database_mapping):
    """
    Execute cross-database SQL using SQLite ATTACH DATABASE.
    Includes table validation to ensure tables exist in the corresponding databases.
    """
    if len(databases) < 2:
        # Execute directly when only one database is involved
        db_path = os.path.join(database_dir, databases[0], f"{databases[0]}.db")
        if os.path.exists(db_path):
            return execute_sql_on_database(cross_db_sql, db_path)
        return None, False

    # Create a temporary database as the main database
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    temp_db_path = temp_db.name

    try:
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # ATTACH all involved databases
        db_aliases = {}
        db_tables_cache = {}  # Cache table names per database
        for i, db_name in enumerate(databases):
            db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
            if os.path.exists(db_path):
                # Use database name as alias (special characters must be handled)
                alias = f"db{i}"
                db_aliases[db_name] = alias
                cursor.execute(f'ATTACH DATABASE "{db_path}" AS {alias}')
                # Cache table names
                db_tables_cache[alias] = get_tables_in_database(db_path)

        # Convert SQL: replace "database_name"."table_name" with "alias"."table_name"
        converted_sql = cross_db_sql
        for db_name, alias in db_aliases.items():
            # Handle "database_name"."table_name"."column_name" first (avoid interference from later replacements)
            pattern2 = rf'"{re.escape(db_name)}"\."([^"]+)"\."([^"]+)"'
            replacement2 = rf'"{alias}"."\1"."\2"'
            converted_sql = re.sub(pattern2, replacement2, converted_sql)

            # Then handle "database_name"."table_name"
            pattern = rf'"{re.escape(db_name)}"\."([^"]+)"'
            replacement = rf'"{alias}"."\1"'
            converted_sql = re.sub(pattern, replacement, converted_sql)

            # Handle "database_name.table_name" (dot-separated, unquoted)
            pattern3 = rf'"{re.escape(db_name)}\.([^"]+)"'
            replacement3 = rf'"{alias}"."\1"'
            converted_sql = re.sub(pattern3, replacement3, converted_sql)

            # Handle "database_name.table_name"."column_name"
            pattern4 = rf'"{re.escape(db_name)}\.([^"]+)"\."([^"]+)"'
            replacement4 = rf'"{alias}"."\1"."\2"'
            converted_sql = re.sub(pattern4, replacement4, converted_sql)

        # Validate table existence after conversion
        table_pattern = rf'"(db\d+)"\."([^"]+)"'
        tables_in_sql = re.findall(table_pattern, converted_sql)
        missing_tables = []
        for alias, table_name in tables_in_sql:
            if alias in db_tables_cache:
                if table_name not in db_tables_cache[alias]:
                    missing_tables.append(f"{alias}.{table_name}")

        if missing_tables:
            # Return failure when tables do not exist
            conn.close()
            os.unlink(temp_db_path)
            return None, False

        # Execute SQL
        cursor.execute(converted_sql)
        results = cursor.fetchall()
        conn.close()

        # Clean up temporary file
        os.unlink(temp_db_path)

        return results, True

    except Exception as e:
        # Clean up temporary file
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
        return None, False

def load_multiple_schemas(database_names, database_dir):
    """Load schema information from multiple databases."""
    schemas = {}
    for db_name in database_names:
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            schemas[db_name] = load_schema(schema_file)
        else:
            print(f"Warning: schema file not found {schema_file}")
    return schemas

def load_cross_database_graph(graph_file):
    """Load a cross-database graph file."""
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    return graph_data

def extract_tables_from_cross_database_graph(graph_data, table_database_mapping):
    """Extract relevant tables from a cross-database graph."""
    tables = set()
    for node in graph_data.get('nodes', []):
        node_type = node.get('node_type')
        if node_type == 'table':
            table_name = node.get('table_name', '')
            if table_name in table_database_mapping:
                # Format: database_name.table_name
                db_name = table_database_mapping[table_name]
                tables.add(f"{db_name}.{table_name}")
    return list(tables)

def extract_columns_from_cross_database_graph(graph_data, table_database_mapping):
    """Extract relevant columns from a cross-database graph."""
    columns_by_table = defaultdict(list)
    for node in graph_data.get('nodes', []):
        node_type = node.get('node_type')
        if node_type == 'column':
            table_name = node.get('table_name', '')
            column_name = node.get('column_name', '')
            if table_name in table_database_mapping:
                db_name = table_database_mapping[table_name]
                full_table_name = f"{db_name}.{table_name}"
                columns_by_table[full_table_name].append(column_name)
    return dict(columns_by_table)

def validate_tables_exist_in_databases(selected_tables, schemas, database_dir):
    """Verify tables exist in the corresponding databases; return only existing tables."""
    valid_tables = []
    for table_full_name in selected_tables:
        # Parse table name in the format "database_name.table_name"
        parts = table_full_name.split('.', 1)
        if len(parts) == 2:
            db_name, table_name = parts
            # Check whether the table is in the schema
            if db_name in schemas:
                schema = schemas[db_name]
                # Check whether the table is in the schema tables list
                table_exists = False
                for table_info in schema.get('tables', []):
                    if table_info.get('table_name') == table_name:
                        table_exists = True
                        break

                # If present in schema, verify it actually exists in the database
                if table_exists:
                    db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
                    if os.path.exists(db_path):
                        tables_in_db = get_tables_in_database(db_path)
                        if table_name in tables_in_db:
                            valid_tables.append(table_full_name)
    return valid_tables

def get_all_tables_from_databases(schemas, database_dir, max_tables_per_db=50):
    """
    Get tables that actually exist across all databases (limit count to avoid overly long prompts).
    Key improvement: query real table names directly from database files instead of reading from schema.
    """
    all_tables = {}
    for db_name, schema in schemas.items():
        db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
        if os.path.exists(db_path):
            # Query real table names directly from the database file (this is critical)
            tables_in_db = get_tables_in_database(db_path)

            all_tables[db_name] = []
            count = 0

            # Use only tables that actually exist in the database
            for table_name in tables_in_db:
                if count >= max_tables_per_db:
                    break

                # Look up corresponding table info in schema (for descriptions and column info)
                table_info_from_schema = None
                for table_info in schema.get('tables', []):
                    if table_info.get('table_name') == table_name:
                        table_info_from_schema = table_info
                        break

                # If not in schema, try matching similar table names (table names may have numeric suffixes)
                if table_info_from_schema is None:
                    # Try matching by stripping numeric suffix
                    base_name = table_name.rsplit('-', 1)[0] if '-' in table_name else table_name
                    for table_info in schema.get('tables', []):
                        schema_table_name = table_info.get('table_name', '')
                        schema_base_name = schema_table_name.rsplit('-', 1)[0] if '-' in schema_table_name else schema_table_name
                        if base_name == schema_base_name:
                            table_info_from_schema = table_info
                            break

                # Get column info (query directly from database for accuracy)
                columns_in_db = []
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(f'PRAGMA table_info("{table_name}")')
                    columns_in_db = [row[1] for row in cursor.fetchall()]
                    conn.close()
                except:
                    # Fall back to column info from schema if query fails
                    if table_info_from_schema:
                        columns_in_db = [col.get('column_name', '') for col in table_info_from_schema.get('columns', [])]

                all_tables[db_name].append({
                    'name': table_name,  # Use the table name that actually exists in the database
                    'description': table_info_from_schema.get('table_description', '') if table_info_from_schema else '',
                    'comment': table_info_from_schema.get('table_comment', '') if table_info_from_schema else '',
                    'columns': columns_in_db[:15]  # Show only the first 15 columns for accuracy
                })
                count += 1
    return all_tables

def extract_compact_graph_info(graph_data, table_database_mapping, schemas, max_tables=20, max_columns_per_table=10):
    """
    Extract compact key information from graph files for prompt construction.

    Extract only:
    1. Tables and columns related to placeholders (most relevant)
    2. Foreign key relationships (used to determine whether JOIN is possible)
    3. Brief table information (limited count and column count)

    Args:
        graph_data: Full graph data
        table_database_mapping: Mapping from table to database
        schemas: Database schemas
        max_tables: Maximum number of tables to extract
        max_columns_per_table: Maximum number of columns to show per table

    Returns:
        dict: {
            'suggested_tables': [...],  # Suggested table list
            'foreign_keys': [...],      # Foreign key relationship list
            'table_info': {...}         # Brief table information
        }
    """
    # 1. Extract placeholder-related tables (from table_database_mapping)
    suggested_tables = []
    for table_name, db_name in table_database_mapping.items():
        table_full_name = f"{db_name}.{table_name}"
        suggested_tables.append(table_full_name)

    # Limit table count
    suggested_tables = suggested_tables[:max_tables]

    # 2. Extract foreign key relationships (only those involving suggested tables)
    foreign_keys = []
    suggested_table_set = set(suggested_tables)

    for edge in graph_data.get('edges', []):
        if edge.get('edge_type') == 'foreign_key':
            source = edge.get('source', '')
            target = edge.get('target', '')

            # Check whether suggested tables are involved
            source_table = '.'.join(source.split('.')[:2]) if '.' in source else None
            target_table = '.'.join(target.split('.')[:2]) if '.' in target else None

            if source_table in suggested_table_set or target_table in suggested_table_set:
                foreign_keys.append({
                    'source': source,
                    'target': target
                })

    # 3. Extract brief table information (only suggested tables, limited columns)
    table_info = {}
    for table_full_name in suggested_tables:
        parts = table_full_name.split('.', 1)
        if len(parts) == 2:
            db_name, table_name = parts
            if db_name in schemas:
                schema = schemas[db_name]
                for table_info_item in schema.get('tables', []):
                    if table_info_item.get('table_name') == table_name:
                        # Extract only the first N columns
                        columns = table_info_item.get('columns', [])[:max_columns_per_table]
                        table_info[table_full_name] = {
                            'description': table_info_item.get('table_description', ''),
                            'comment': table_info_item.get('table_comment', ''),
                            'columns': [
                                {
                                    'name': col.get('column_name', ''),
                                    'type': col.get('data_type', 'TEXT')
                                }
                                for col in columns
                            ],
                            'total_columns': len(table_info_item.get('columns', []))
                        }
                        break

    return {
        'suggested_tables': suggested_tables,
        'foreign_keys': foreign_keys,
        'table_info': table_info
    }

def construct_cross_database_prompt_join(sql_skeleton, schemas, table_database_mapping,
                                   graph_data, sql_analysis, database_dir, recommended_join_columns=None):
    """Build cross-database SQL filling prompt - JOIN version: emphasizes JOIN, aggregate functions, and complex query structure."""

    # Use compact graph info extraction (key information only)
    compact_info = extract_compact_graph_info(
        graph_data,
        table_database_mapping,
        schemas,
        max_tables=20,  # At most 20 suggested tables
        max_columns_per_table=10  # At most 10 columns per table
    )

    suggested_tables = compact_info['suggested_tables']
    foreign_keys = compact_info['foreign_keys']
    table_info = compact_info['table_info']

    # Verify suggested tables exist in the corresponding databases
    valid_suggested_tables = validate_tables_exist_in_databases(suggested_tables, schemas, database_dir)

    # Get tables that actually exist across all databases (give the LLM more choices, but limit count)
    all_available_tables = get_all_tables_from_databases(schemas, database_dir, max_tables_per_db=20)

    # Build detailed info for suggested tables (show only suggested tables, compact format)
    suggested_tables_info = ""
    if valid_suggested_tables:
        suggested_tables_info = "\nSuggested tables (from SQL skeleton analysis, for reference only; you may choose other more suitable tables):\n"
        for table_full_name in valid_suggested_tables[:15]:  # Show at most 15
            if table_full_name in table_info:
                info = table_info[table_full_name]
                suggested_tables_info += f"\n  - {table_full_name}\n"
                if info.get('description'):
                    suggested_tables_info += f"    Description: {info['description'][:100]}...\n"  # Limit description length
                if info.get('comment'):
                    suggested_tables_info += f"    Comment: {info['comment'][:100]}...\n"
                suggested_tables_info += f"    Columns (first {len(info['columns'])}, total {info['total_columns']}):\n"
                for col in info['columns']:
                    suggested_tables_info += f"      - {col['name']} ({col['type']})\n"

    # Build brief info for all available tables (table names and key info only, heavily compressed)
    # Important: table names shown here are queried directly from database files to ensure accuracy
    all_tables_info = ""
    for db_name, tables_list in all_available_tables.items():
        all_tables_info += f"\nDatabase: {db_name} ({len(tables_list)} tables total, showing first 15; **these table names are real names queried directly from the database, guaranteed to exist**)\n"
        for table in tables_list[:15]:  # Show at most 15 tables per database
            table_name = table['name']
            # Show table name and column count (so the LLM knows table structure)
            column_count = len(table.get('columns', []))
            all_tables_info += f"  - {db_name}.{table_name} ({column_count} columns)\n"
            # Show first 5 column names (help the LLM choose correct columns)
            if table.get('columns'):
                all_tables_info += f"    Columns (first 5): {', '.join(table['columns'][:5])}\n"

    # Build foreign key relationship info (only foreign keys involving suggested tables)
    fk_text = ""
    fk_count = 0
    for fk in foreign_keys[:20]:  # Show at most 20 foreign key relationships
        source = fk.get('source', '')
        target = fk.get('target', '')
        if source and target:
            fk_text += f"  - {source} -> {target}\n"
            fk_count += 1

    # Check whether foreign key relationships exist (for JOIN)
    has_foreign_keys = fk_count > 0

    # Format recommended JOIN column pairs
    recommended_join_text = ""
    if recommended_join_columns:
        join_lines = [f"  - {k} = {v}" for k, v in recommended_join_columns.items()]
        recommended_join_text = "**Recommended JOIN column pairs (from table-pair analysis, strongly recommended)**:\n" + "\n".join(join_lines)

    # SQL skeleton analysis hints (JOIN version: emphasizes JOIN, aggregate functions, etc.)
    analysis_hints = ""
    analysis_hints += "\n**Important: This version generates cross-database SQL using JOIN, emphasizing complex query structure!**\n"

    if sql_analysis['has_join']:
        analysis_hints += "\n✅ **This SQL skeleton contains JOIN operations; generate SQL using JOIN:**\n"
        analysis_hints += "  - **Must use JOIN**; do not convert to UNION\n"
        analysis_hints += "  - Prefer JOIN via foreign key relationships (if available)\n"
        analysis_hints += "  - If no foreign keys, use semantically related columns for JOIN (e.g., name, ID)\n"
        analysis_hints += "  - JOIN example:\n"
        analysis_hints += "    SELECT \"db1\".\"table1\".\"col1\", \"db2\".\"table2\".\"col2\"\n"
        analysis_hints += "    FROM \"db1\".\"table1\"\n"
        analysis_hints += "    JOIN \"db2\".\"table2\" ON \"db1\".\"table1\".\"join_col\" = \"db2\".\"table2\".\"join_col\"\n"
        analysis_hints += "    WHERE \"db1\".\"table1\".\"col1\" IS NOT NULL\n"

    if sql_analysis['has_aggregate']:
        analysis_hints += "\n✅ **This SQL skeleton contains aggregate functions; keep and use them:**\n"
        analysis_hints += "  - **Must use aggregate functions** (COUNT, SUM, AVG, MAX, MIN, etc.)\n"
        analysis_hints += "  - Use GROUP BY for grouping\n"
        analysis_hints += "  - Example: SELECT \"db1\".\"table1\".\"group_col\", COUNT(*) AS count FROM ... GROUP BY \"db1\".\"table1\".\"group_col\"\n"
    else:
        analysis_hints += "\n💡 **Consider adding aggregate functions to increase query complexity:**\n"
        analysis_hints += "  - You may use COUNT, SUM, AVG, and other aggregate functions\n"
        analysis_hints += "  - Use GROUP BY for grouped statistics\n"
        analysis_hints += "  - Use ORDER BY to sort results\n"

    if sql_analysis['has_subquery']:
        analysis_hints += "\n✅ **This SQL skeleton contains subqueries; you may keep the subquery structure:**\n"
        analysis_hints += "  - Subqueries can execute in cross-database scenarios (using ATTACH DATABASE)\n"
        analysis_hints += "  - Ensure table and column names in subqueries are correct\n"
    else:
        analysis_hints += "\n💡 **Consider using subqueries to increase complexity:**\n"
        analysis_hints += "  - Use IN subquery: WHERE col IN (SELECT col FROM \"db2\".\"table2\" WHERE ...)\n"
        analysis_hints += "  - Use EXISTS subquery: WHERE EXISTS (SELECT 1 FROM \"db2\".\"table2\" WHERE ...)\n"

    # General suggestions (JOIN version)
    analysis_hints += "\n**General suggestions (JOIN version):**\n"
    analysis_hints += "  - **Must use JOIN**; do not use UNION\n"
    analysis_hints += "  - Prefer JOIN via foreign key relationships (see foreign key list above)\n"
    analysis_hints += "  - If no foreign keys, use semantically related columns (e.g., name, ID, code) for JOIN\n"
    analysis_hints += "  - **Encourage aggregate functions** (COUNT, SUM, AVG, etc.) and GROUP BY\n"
    analysis_hints += "  - **Encourage ORDER BY** to sort results\n"
    analysis_hints += "  - You may use HAVING to filter grouped results\n"
    analysis_hints += "  - Choose tables with data to ensure the query returns results\n"

    # Build complete prompt
    databases_str = ', '.join(schemas.keys())

    prompt = f"""Based on the following SQL framework and database information, generate a complete cross-database SQL statement that can execute correctly on SQLite.

**Important: This is a cross-database query involving the following databases: {databases_str}**

**Core principle: This version generates cross-database SQL using JOIN, emphasizing complex query structure and diversity!**

**Important: This version must use JOIN and encourages aggregate functions, GROUP BY, ORDER BY, and other complex structures:**
1. **Must use JOIN**:
   - **UNION is prohibited**; must use JOIN to connect tables from different databases
   - Prefer JOIN via foreign key relationships (see foreign key list below)
   - If no foreign keys, use semantically related columns for JOIN (e.g., name, ID, code)
   - JOIN example: FROM "db1"."table1" JOIN "db2"."table2" ON "db1"."table1"."join_col" = "db2"."table2"."join_col"

2. **Encourage aggregate functions and GROUP BY**:
   - Use COUNT, SUM, AVG, MAX, MIN, and other aggregate functions
   - Use GROUP BY for grouped statistics
   - Example: SELECT "db1"."table1"."group_col", COUNT(*) AS count FROM ... GROUP BY "db1"."table1"."group_col"
   - You may use HAVING to filter grouped results

3. **Encourage ORDER BY sorting**:
   - Use ORDER BY to sort results (ASC or DESC)
   - You may sort by multiple columns
   - Example: ORDER BY "db1"."table1"."col1" DESC, "db2"."table2"."col2" ASC

4. **Subqueries are allowed**:
   - Use IN subquery: WHERE col IN (SELECT col FROM "db2"."table2" WHERE ...)
   - Use EXISTS subquery: WHERE EXISTS (SELECT 1 FROM "db2"."table2" WHERE ...)
   - Ensure table and column names in subqueries are correct

5. **Table selection**:
   - Prefer tables with foreign key relationships (see foreign key list below)
   - If no foreign keys, choose semantically related tables (e.g., enterprise info and credit info tables)
   - Ensure selected tables actually exist in the corresponding databases

**Strict requirements:**
- **Output only the final complete SQL statement; do not repeat the prompt content.**
- **Generated SQL must be syntactically correct and runnable on SQLite to produce results.**
- **Do not add any extra explanation, comments, or output formatting (code blocks, etc.).**
- **⚠️ Critical: Must use real table and column names listed in "All available tables" below; these are queried directly from databases and guaranteed to exist!**
- **⚠️ Never use table or column names that exist in schema but not in the actual database!**
- **All table names must use format: "database_name"."table_name", e.g.: "企业服务"."市市场监管局-市场主体注册情况-1820"**
- **All column names must use format: "database_name"."table_name"."column_name", e.g.: "企业服务"."市市场监管局-市场主体注册情况-1820"."市场主体名称"**
- **All table and column names must be wrapped in double quotes (including Chinese and special characters).**
- **⚠️ Table names must exactly match those listed in "All available tables" (including numeric suffixes); do not omit or modify!**

**SQL framework (for reference only; you may simplify or change structure, prioritizing executability):**
{sql_skeleton}

{suggested_tables_info}

All available tables (choose the most suitable ones):
{all_tables_info}

Foreign key relationships (**prefer these for JOIN**):
{fk_text if fk_text else "⚠️ No foreign key relationships - cross-database queries may lack foreign keys. Use semantically related columns (e.g., name, ID, code) for JOIN."}

{recommended_join_text}

{analysis_hints}

Generate the complete cross-database SQL statement (you may optimize and adjust the SQL skeleton as needed):"""

    # Return all available tables (for subsequent validation)
    all_tables_list = []
    for db_name, tables_list in all_available_tables.items():
        for table in tables_list:
            all_tables_list.append(f"{db_name}.{table['name']}")

    return prompt, all_tables_list, {}

def process_cross_database_skeleton(skeleton_data, schemas, graph_dir, output_dir,
                                    database_dir, max_retries=3):
    """Process a single cross-database SQL skeleton and fill it to generate complete SQL."""

    sql_skeleton = skeleton_data['sql_skeleton']
    table_database_mapping = skeleton_data['table_database_mapping']
    databases = skeleton_data.get('databases', [])

    # Determine output filename
    original_file = skeleton_data.get('original_file', 'unknown')
    match = re.search(r'(\d+)', original_file)
    if match:
        idx = match.group(1)
        output_file = os.path.join(output_dir, f"cross_db_generated_sql_{idx}.json")
    else:
        import hashlib
        hash_id = hashlib.md5(sql_skeleton.encode()).hexdigest()[:8]
        output_file = os.path.join(output_dir, f"cross_db_generated_sql_{hash_id}.json")

    # Check whether output already exists (regenerate after modifications, so do not skip)
    # if os.path.exists(output_file):
    #     return idx if match else hash_id, True, "already exists"

    # Load graph file
    graph_file = os.path.join(graph_dir, f"cross_db_graph_{idx if match else hash_id}.json")
    if not os.path.exists(graph_file):
        return idx if match else hash_id, False, "Graph file not found"

    graph_data = load_cross_database_graph(graph_file)

    # Analyze SQL skeleton
    sql_analysis = analyze_sql_skeleton(sql_skeleton)

    # Get recommended JOIN column pairs (if available)
    recommended_join_columns = skeleton_data.get('recommended_join_columns', None)

    # Build prompt (JOIN version)
    prompt, selected_tables, selected_columns = construct_cross_database_prompt_join(
        sql_skeleton, schemas, table_database_mapping, graph_data, sql_analysis, database_dir,
        recommended_join_columns=recommended_join_columns
    )

    # Do not skip; even without suggested tables, provide all available tables for the LLM to choose from
    if prompt is None:
        # If construction fails, use fallback: provide all available tables
        all_available_tables = get_all_tables_from_databases(schemas, database_dir)
        if not any(all_available_tables.values()):
            return None, False, "No available tables in database"
        # Rebuild prompt using all available tables (JOIN version)
        recommended_join_columns = skeleton_data.get('recommended_join_columns', None)
        prompt, _, _ = construct_cross_database_prompt_join(
            sql_skeleton, schemas, table_database_mapping, graph_data, sql_analysis, database_dir,
            recommended_join_columns=recommended_join_columns
        )

    # Call LLM to generate SQL
    client = get_client()
    API_CONFIG = load_config()

    # Ensure base_url format is correct
    api_url = API_CONFIG.get("api_url", "").rstrip('/')
    model = API_CONFIG.get("model", "gpt-4o-mini")  # Use gpt-4o-mini, consistent with user config

    # Initialize variables for storing error information
    generated_sql = None
    results = None
    execution_error = None
    last_error = None
    final_attempt = 0

    for attempt in range(max_retries):
        final_attempt = attempt + 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a SQL expert skilled at generating cross-database SQL queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=API_CONFIG.get("temperature", 0.1),
                max_tokens=API_CONFIG.get("max_tokens", 8000)
            )

            generated_sql = response.choices[0].message.content.strip()

            # Clean SQL (remove code block markers, etc.)
            generated_sql = re.sub(r'^```sql\s*', '', generated_sql, flags=re.IGNORECASE)
            generated_sql = re.sub(r'^```\s*', '', generated_sql)
            generated_sql = re.sub(r'```\s*$', '', generated_sql)
            generated_sql = generated_sql.strip()

            # Validate SQL syntax
            try:
                sqlparse.parse(generated_sql)
            except Exception as parse_error:
                last_error = f"SQL syntax error: {str(parse_error)}"
                if attempt < max_retries - 1:
                    continue
                # Last attempt failed; save file and return
                break

            # Execute SQL and get results
            # Use ATTACH DATABASE to execute cross-database SQL
            results = None
            execution_error = None

            try:
                # Method 1: use ATTACH DATABASE to execute true cross-database queries
                results, success = execute_cross_database_sql_with_attach(
                    generated_sql, databases, database_dir, table_database_mapping
                )

                if not success:
                    # Method 2: if ATTACH fails, try converting to single-database format (fallback)
                    single_db_sql = convert_to_single_database_sql(generated_sql, table_database_mapping)

                    # Try executing on involved databases (prefer the first database)
                    for db_name in databases:
                        db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
                        if os.path.exists(db_path):
                            results, success = execute_sql_on_database(single_db_sql, db_path)
                            if success:
                                break  # Successful execution; exit loop

                    # If all methods fail, record the error
                    if not success:
                        execution_error = "Cannot execute SQL on any database (both ATTACH and single-database format failed)"
                        results = None
                else:
                    # ATTACH succeeded; results may be an empty list (empty results still count as success)
                    if results is None:
                        results = []
            except Exception as e:
                execution_error = f"Execution exception: {str(e)}"

            # At this point, API call and SQL generation succeeded; save results
            saved_results = []
            if results is not None:
                # Save only the first 10 results (consistent with single-database behavior)
                saved_results = results[:10] if len(results) > 10 else results
                # Convert to list format (ensure JSON serializable)
                saved_results = [list(row) for row in saved_results]

            # Save results (save regardless of execution errors)
            result = {
                'sql': generated_sql,
                'results': saved_results,
                'sql_skeleton': sql_skeleton,
                'databases': databases,
                'table_database_mapping': table_database_mapping,
                'tables': selected_tables,
                'columns': selected_columns,
                'metadata': {
                    'has_join': sql_analysis['has_join'],
                    'has_subquery': sql_analysis['has_subquery'],
                    'has_aggregate': sql_analysis['has_aggregate'],
                    'is_cross_database': True,
                    'num_databases': len(databases)
                },
                'generation_info': {
                    'model': model,
                    'attempt': final_attempt
                }
            }

            # Record execution errors in metadata if present
            if execution_error:
                result['metadata']['execution_error'] = execution_error

            # Save file (save whether successful or failed)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            # Determine success (has results and no execution error)
            if results is not None and len(saved_results) > 0 and not execution_error:
                return idx if match else hash_id, True, "Success"
            else:
                return idx if match else hash_id, False, execution_error or "Execution returned no results"

        except Exception as api_error:
            # Record API error
            error_str = str(api_error)
            last_error = f"API call failed: {error_str[:200]}"

            # Check for quota errors
            if "quota" in error_str.lower() or "429" in error_str or "insufficient_quota" in error_str:
                last_error = f"API quota exhausted: {error_str[:200]}"
                # Do not retry quota errors; save and return immediately
                break
            # Retry other API errors
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

    # If all attempts fail, save error information to file
    result = {
        'sql': generated_sql or "",
        'results': None,
        'sql_skeleton': sql_skeleton,
        'databases': databases,
        'table_database_mapping': table_database_mapping,
        'tables': selected_tables if 'selected_tables' in locals() else [],
        'columns': selected_columns if 'selected_columns' in locals() else [],
        'metadata': {
            'has_join': sql_analysis['has_join'],
            'has_subquery': sql_analysis['has_subquery'],
            'has_aggregate': sql_analysis['has_aggregate'],
            'is_cross_database': True,
            'num_databases': len(databases),
            'error': last_error or "Reached maximum retry count",
            'execution_error': execution_error
        },
        'generation_info': {
            'model': model,
            'attempt': final_attempt
        }
    }

    # Save failed file (for easier debugging)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return idx if match else hash_id, False, last_error or "Reached maximum retry count"

def main():
    parser = argparse.ArgumentParser(description='Fill cross-database SQL skeletons')
    parser.add_argument('--skeleton_file', type=str, required=True,
                       help='Cross-database SQL skeleton file')
    parser.add_argument('--graph_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph',
                       help='Graph file directory')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='Database directory')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='Output directory')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum number of retries')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='Number of concurrent worker threads')

    args = parser.parse_args()

    # Load cross-database SQL skeletons
    print(f"Loading cross-database SQL skeletons: {args.skeleton_file}")
    with open(args.skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)

    print(f"Total SQL skeletons: {len(skeletons)}")

    # Collect all involved databases
    all_databases = set()
    for skeleton in skeletons:
        all_databases.update(skeleton.get('databases', []))

    print(f"Involved databases: {sorted(all_databases)}")

    # Load schemas from all databases
    print("\nLoading database schemas...")
    schemas = load_multiple_schemas(all_databases, args.database_dir)
    print(f"Successfully loaded schemas from {len(schemas)} databases")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each SQL skeleton
    print(f"\nFilling SQL skeletons...")
    success_count = 0
    failed_count = 0

    # Process concurrently with a thread pool
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for skeleton in skeletons:
            future = executor.submit(
                process_cross_database_skeleton,
                skeleton, schemas, args.graph_dir, args.output_dir,
                args.database_dir, args.max_retries
            )
            futures.append(future)

        # Collect results
        for future in tqdm(as_completed(futures), total=len(futures), desc="Filling progress"):
            idx, success, message = future.result()
            if success:
                success_count += 1
            else:
                failed_count += 1
                if failed_count <= 10:  # Show only the first 10 errors
                    print(f"\nFailed (idx={idx}): {message}")

    print(f"\nDone!")
    print(f"Success: {success_count}/{len(skeletons)}")
    print(f"Failed: {failed_count}/{len(skeletons)}")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()
