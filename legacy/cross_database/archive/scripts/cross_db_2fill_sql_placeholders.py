#!/usr/bin/env python3
"""
Cross-database SQL skeleton fill script

Based on the single-database SQL fill script, extended for cross-database scenarios:
1. Load schemas for multiple databases and graph files
2. Explicitly inform the LLM in the prompt that this is a cross-database query
3. Let the LLM generate SQL with database prefixes (e.g., db_name.table_name)
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
    os.path.join(sql_filling_dir, "2fill_sql_placeholders_improved.py")
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)

load_config = fill_module.load_config
get_client = fill_module.get_client
load_schema = fill_module.load_schema
analyze_sql_skeleton = fill_module.analyze_sql_skeleton
construct_enhanced_prompt = fill_module.construct_enhanced_prompt
# execute_sql may not exist; cross-database SQL doesn't need direct execution validation
load_graph_metadata = getattr(fill_module, 'load_graph_metadata', None)

def convert_to_single_database_sql(cross_db_sql, table_database_mapping):
    """Convert cross-database SQL to single-database SQL (remove database prefix)"""
    single_db_sql = cross_db_sql
    # Replace "db_name"."table_name" with "table_name"
    for table, db in table_database_mapping.items():
        # Handle quoted case
        pattern1 = rf'"{re.escape(db)}"\."{re.escape(table)}"'
        replacement1 = f'"{table}"'
        single_db_sql = re.sub(pattern1, replacement1, single_db_sql)
        
        # Handle unquoted case
        pattern2 = rf'{re.escape(db)}\.{re.escape(table)}'
        replacement2 = table
        single_db_sql = re.sub(pattern2, replacement2, single_db_sql)
    
    # Replace "db_name"."table_name"."col_name" with "table_name"."col_name"
    for table, db in table_database_mapping.items():
        pattern = rf'"{re.escape(db)}"\."{re.escape(table)}"\."([^"]+)"'
        replacement = rf'"{table}"."\1"'
        single_db_sql = re.sub(pattern, replacement, single_db_sql)
    
    return single_db_sql

def execute_sql_on_database(sql, db_path):
    """Execute SQL on single database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        # Empty results also count as success (consistent with single-database)
        if results:
            return results, True
        else:
            return [], True  # Empty results also count as success
    except sqlite3.Error as e:
        return None, False
    except Exception as e:
        return None, False

def get_tables_in_database(db_path, alias=None):
    """Get all table names in database"""
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
    """Validate tables used in SQL exist in corresponding database"""
    # Extract table names used in SQL
    table_pattern = r'"(?:db\d+|[\u4e00-\u9fa5]+)"\."([^"]+)"'
    tables_in_sql = set(re.findall(table_pattern, sql))
    
    # Check each table exists in corresponding database
    missing_tables = []
    for table_name in tables_in_sql:
        # Find database corresponding to table
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
    Execute cross-database SQL using SQLite ATTACH DATABASE
    Add table name validation, ensure tables exist in corresponding database
    """
    if len(databases) < 2:
        # If only one database, execute directly
        db_path = os.path.join(database_dir, databases[0], f"{databases[0]}.db")
        if os.path.exists(db_path):
            return execute_sql_on_database(cross_db_sql, db_path)
        return None, False
    
    # Create temporary database as main database
    import tempfile
    temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    temp_db.close()
    temp_db_path = temp_db.name
    
    try:
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # ATTACH all involved databases
        db_aliases = {}
        db_tables_cache = {}  # Cache table names for each database
        for i, db_name in enumerate(databases):
            db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
            if os.path.exists(db_path):
                # Use database name as alias (but handle special characters)
                alias = f"db{i}"
                db_aliases[db_name] = alias
                cursor.execute(f'ATTACH DATABASE "{db_path}" AS {alias}')
                # Cache table names
                db_tables_cache[alias] = get_tables_in_database(db_path)
        
        # Convert SQL: "db_name"."table_name" to "alias"."table_name"
        converted_sql = cross_db_sql
        for db_name, alias in db_aliases.items():
            # Process "database name"."table name"."column name" format first (avoid interference from later replacements)
            pattern2 = rf'"{re.escape(db_name)}"\."([^"]+)"\."([^"]+)"'
            replacement2 = rf'"{alias}"."\1"."\2"'
            converted_sql = re.sub(pattern2, replacement2, converted_sql)
            
            # Then process "database name"."table name" format
            pattern = rf'"{re.escape(db_name)}"\."([^"]+)"'
            replacement = rf'"{alias}"."\1"'
            converted_sql = re.sub(pattern, replacement, converted_sql)
            
            # Process "database name.table name" format (dot-separated, without quotes)
            pattern3 = rf'"{re.escape(db_name)}\.([^"]+)"'
            replacement3 = rf'"{alias}"."\1"'
            converted_sql = re.sub(pattern3, replacement3, converted_sql)
            
            # Process "database name.table name"."column name" format
            pattern4 = rf'"{re.escape(db_name)}\.([^"]+)"\."([^"]+)"'
            replacement4 = rf'"{alias}"."\1"."\2"'
            converted_sql = re.sub(pattern4, replacement4, converted_sql)
        
        # Validate table exists (validate after conversion)
        table_pattern = rf'"(db\d+)"\."([^"]+)"'
        tables_in_sql = re.findall(table_pattern, converted_sql)
        missing_tables = []
        for alias, table_name in tables_in_sql:
            if alias in db_tables_cache:
                if table_name not in db_tables_cache[alias]:
                    missing_tables.append(f"{alias}.{table_name}")
        
        if missing_tables:
            # Table does not exist; return failure
            conn.close()
            os.unlink(temp_db_path)
            return None, False
        
        # Execute SQL
        cursor.execute(converted_sql)
        results = cursor.fetchall()
        conn.close()
        
        # Clean up temp files
        os.unlink(temp_db_path)
        
        return results, True
        
    except Exception as e:
        # Clean up temp files
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
        return None, False

def load_multiple_schemas(database_names, database_dir):
    """Load schema info for multiple databases"""
    schemas = {}
    for db_name in database_names:
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            schemas[db_name] = load_schema(schema_file)
        else:
            print(f"Warning: Schema file not found {schema_file}")
    return schemas

def load_cross_database_graph(graph_file):
    """Load cross-database graph file"""
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    return graph_data

def extract_tables_from_cross_database_graph(graph_data, table_database_mapping):
    """Extract relevant tables from the cross-database graph."""
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
    """Extract relevant columns from the cross-database graph."""
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
    """Validate tables exist in corresponding database, return only existing tables"""
    valid_tables = []
    for table_full_name in selected_tables:
        # Parse table name in "database.table" format
        parts = table_full_name.split('.', 1)
        if len(parts) == 2:
            db_name, table_name = parts
            # Check whether the table exists in the schema
            if db_name in schemas:
                schema = schemas[db_name]
                # Check whether the table exists in the schema tables list
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
    Get all real tables across databases (limit count to avoid overly long prompt).

    Key improvement: query real table names directly from database files, not from schema.
    """
    all_tables = {}
    for db_name, schema in schemas.items():
        db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
        if os.path.exists(db_path):
            # Query real table names directly from database files (critical!)
            tables_in_db = get_tables_in_database(db_path)
            
            all_tables[db_name] = []
            count = 0
            
            # Only use tables that actually exist in database
            for table_name in tables_in_db:
                if count >= max_tables_per_db:
                    break
                
                # Look up corresponding table info from schema (for descriptions and column info)
                table_info_from_schema = None
                for table_info in schema.get('tables', []):
                    if table_info.get('table_name') == table_name:
                        table_info_from_schema = table_info
                        break
                
                # If not found in schema, try matching similar table names (may have suffix numbers)
                if table_info_from_schema is None:
                    # Try matching by stripping suffix numbers
                    base_name = table_name.rsplit('-', 1)[0] if '-' in table_name else table_name
                    for table_info in schema.get('tables', []):
                        schema_table_name = table_info.get('table_name', '')
                        schema_base_name = schema_table_name.rsplit('-', 1)[0] if '-' in schema_table_name else schema_table_name
                        if base_name == schema_base_name:
                            table_info_from_schema = table_info
                            break
                
                # Get column info (query directly from database, ensuring accuracy)
                columns_in_db = []
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(f'PRAGMA table_info("{table_name}")')
                    columns_in_db = [row[1] for row in cursor.fetchall()]
                    conn.close()
                except:
                    # If query fails, fall back to column info from schema
                    if table_info_from_schema:
                        columns_in_db = [col.get('column_name', '') for col in table_info_from_schema.get('columns', [])]
                
                all_tables[db_name].append({
                    'name': table_name,  # Use real table names from database
                    'description': table_info_from_schema.get('table_description', '') if table_info_from_schema else '',
                    'comment': table_info_from_schema.get('table_comment', '') if table_info_from_schema else '',
                    'columns': columns_in_db[:15]  # Show only first 15 columns, ensure accuracy
                })
                count += 1
    return all_tables

def extract_compact_graph_info(graph_data, table_database_mapping, schemas, max_tables=20, max_columns_per_table=10):
    """
    Extract compressed key info from graph files for building prompt
    
    Extract only:
    1. Placeholder-related tables and columns (most relevant)
    2. Foreign key relationships (to determine if JOIN possible)
    3. Brief table info (limit count and columns)
    
    Args:
        graph_data: Full graph data
        table_database_mapping: Table-to-database mapping
        schemas: Database schemas
        max_tables: Maximum number of tables to extract
        max_columns_per_table: Maximum columns to show per table
    
    Returns:
        dict: {
            'suggested_tables': [...],  # Suggested table list
            'foreign_keys': [...],      # Foreign key relationship list
            'table_info': {...}         # Brief table info
        }
    """
    # 1. Extract placeholder-related tables (from table_database_mapping)
    suggested_tables = []
    for table_name, db_name in table_database_mapping.items():
        table_full_name = f"{db_name}.{table_name}"
        suggested_tables.append(table_full_name)
    
    # Limit table count
    suggested_tables = suggested_tables[:max_tables]
    
    # 2. Extract foreign key relationships (only FKs involving suggested tables)
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
    
    # 3. Extract brief table info (only suggested tables, limit column count)
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

def construct_cross_database_prompt(sql_skeleton, schemas, table_database_mapping, 
                                   graph_data, sql_analysis, database_dir):
    """Build cross-database SQL fill prompt - optimized version using compressed graph info."""

    # Use compressed graph info extraction (extract only key info)
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
    
    # Validate suggested tables exist in corresponding database
    valid_suggested_tables = validate_tables_exist_in_databases(suggested_tables, schemas, database_dir)
    
    # Get all real tables across databases (give LLM more choices, but limit count）
    all_available_tables = get_all_tables_from_databases(schemas, database_dir, max_tables_per_db=20)
    
    # Build detailed info for suggested tables (show only suggested tables, compressed info)
    suggested_tables_info = ""
    if valid_suggested_tables:
        suggested_tables_info = "\nSuggested tables (from SQL skeleton analysis, for reference only; you may choose other more suitable tables):\n"
        for table_full_name in valid_suggested_tables[:15]:  # Show at most 15
            if table_full_name in table_info:
                info = table_info[table_full_name]
                suggested_tables_info += f"\n  - {table_full_name}\n"
                if info.get('description'):
                    suggested_tables_info += f"    Description: {info['description'][:100]}...\n"
                if info.get('comment'):
                    suggested_tables_info += f"    Comment: {info['comment'][:100]}...\n"
                suggested_tables_info += f"    Columns (first {len(info['columns'])}, total {info['total_columns']}):\n"
                for col in info['columns']:
                    suggested_tables_info += f"      - {col['name']} ({col['type']})\n"
    
    # Build brief info for all available tables (show only table names and key info, heavily compressed）
    # Important: table names shown here are real table names queried directly from database files, ensuring accuracy
    all_tables_info = ""
    for db_name, tables_list in all_available_tables.items():
        all_tables_info += f"\nDatabase: {db_name} ({len(tables_list)} tables total, showing first 15; **these table names are real names queried directly from the database, guaranteed to exist**)\n"
        for table in tables_list[:15]:  # Show at most 15 tables per database
            table_name = table['name']
            column_count = len(table.get('columns', []))
            all_tables_info += f"  - {db_name}.{table_name} ({column_count} columns)\n"
            if table.get('columns'):
                all_tables_info += f"    Columns (first 5): {', '.join(table['columns'][:5])}\n"

    # Build foreign key relationship info (show only FKs involving suggested tables)
    fk_text = ""
    fk_count = 0
    for fk in foreign_keys[:20]:  # Show at most 20 foreign key relationships
        source = fk.get('source', '')
        target = fk.get('target', '')
        if source and target:
            fk_text += f"  - {source} -> {target}\n"
            fk_count += 1
    
    # Check for foreign key relationships (to decide whether to suggest UNION)
    has_foreign_keys = fk_count > 0

    # SQL skeleton analysis hints (emphasize simplification is allowed)
    analysis_hints = ""
    analysis_hints += "\n**Important: You may greatly simplify the SQL skeleton; prioritize executability!**\n"

    if sql_analysis['has_join']:
        analysis_hints += "\n⚠️ **This SQL skeleton contains JOIN operations; consider converting to UNION:**\n"
        analysis_hints += "  - Cross-database JOINs are often hard to execute (missing foreign key relationships)\n"
        analysis_hints += "  - **Strongly recommend UNION**: select semantically related tables from different databases, query separately, then merge\n"
        analysis_hints += "  - UNION example (simple and executable):\n"
        analysis_hints += "    SELECT \"db1\".\"table1\".\"col1\", \"db1\".\"table1\".\"col2\" FROM \"db1\".\"table1\" WHERE \"db1\".\"table1\".\"col1\" IS NOT NULL\n"
        analysis_hints += "    UNION\n"
        analysis_hints += "    SELECT \"db2\".\"table2\".\"col1\", \"db2\".\"table2\".\"col2\" FROM \"db2\".\"table2\" WHERE \"db2\".\"table2\".\"col1\" IS NOT NULL\n"
        analysis_hints += "  - If JOIN is required, keep JOIN conditions simple (e.g., ON table1.col1 = table2.col2) and use real column names\n"

    if sql_analysis['has_aggregate']:
        analysis_hints += "\n⚠️ **This SQL skeleton contains aggregate functions; consider simplifying:**\n"
        analysis_hints += "  - If GROUP BY/COUNT etc. are complex, simplify to a plain SELECT\n"
        analysis_hints += "  - Prefer simple executable queries over complex aggregate queries\n"

    if sql_analysis['has_subquery']:
        analysis_hints += "\n⚠️ **This SQL skeleton contains subqueries; consider simplifying:**\n"
        analysis_hints += "  - Subqueries are hard to execute in cross-database scenarios\n"
        analysis_hints += "  - **Recommend simple SELECT or UNION**; avoid subqueries\n"

    analysis_hints += "\n**General suggestions:**\n"
    analysis_hints += "  - Prefer UNION over JOIN\n"
    analysis_hints += "  - Use simple WHERE conditions (IS NOT NULL, = 'value', etc.)\n"
    analysis_hints += "  - Choose tables with data to ensure the query returns results\n"
    analysis_hints += "  - If the skeleton is too complex, simplify to: SELECT col FROM table WHERE condition\n"

    # Build complete prompt
    databases_str = ', '.join(schemas.keys())

    prompt = f"""Based on the following SQL framework and database information, generate a complete cross-database SQL statement that can execute correctly on SQLite.

**Important: This is a cross-database query involving the following databases: {databases_str}**

**Core principle: Prioritize simple, executable SQL; you may greatly simplify the SQL skeleton!**

**Important: You may modify and simplify the SQL skeleton to make execution easier:**
1. **You may completely change the SQL structure**:
   - If the skeleton is a complex JOIN, convert to simple UNION
   - If the skeleton has subqueries, simplify to single-level queries
   - If the skeleton has aggregates, convert to simple SELECT
   - **Goal is executable SQL, not strict adherence to the skeleton**

2. **Prefer UNION** (recommended):
   - UNION is easier to execute than JOIN
   - Select semantically related tables from different databases, query separately, then merge
   - Ensure each SELECT has compatible column count and types
   - Example: SELECT "db1"."table1"."col1" FROM "db1"."table1" WHERE ... UNION SELECT "db2"."table2"."col1" FROM "db2"."table2" WHERE ...

3. **If JOIN is required**:
   - Ensure JOIN conditions use real existing column names
   - Prefer simple equi-joins (e.g., ON table1.col1 = table2.col2)
   - If JOIN is difficult, switch to UNION immediately

4. **Simplify query conditions**:
   - Use simple WHERE conditions (e.g., IS NOT NULL, = 'value')
   - Avoid complex subqueries
   - Avoid complex aggregate functions

5. **Table selection**:
   - Prefer tables with data (from "All available tables")
   - If suggested tables are unsuitable, freely choose other tables
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

Foreign key relationships:
{fk_text if fk_text else "⚠️ No foreign key relationships - cross-database queries may lack foreign keys. If JOIN is difficult, use UNION to merge query results from different databases."}

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
    """Process a single cross-database SQL skeleton and generate complete SQL."""

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

    # Check if already exists (regenerate after modification, so don't skip)
    # if os.path.exists(output_file):
    #     return idx if match else hash_id, True, "Already exists"

    # Load graph file
    graph_file = os.path.join(graph_dir, f"cross_db_graph_{idx if match else hash_id}.json")
    if not os.path.exists(graph_file):
        return idx if match else hash_id, False, "Graph file does not exist"

    graph_data = load_cross_database_graph(graph_file)

    # Analyze SQL skeleton
    sql_analysis = analyze_sql_skeleton(sql_skeleton)

    # Build prompt
    prompt, selected_tables, selected_columns = construct_cross_database_prompt(
        sql_skeleton, schemas, table_database_mapping, graph_data, sql_analysis, database_dir
    )

    # No longer skip; even without suggested tables, provide all available tables for LLM to choose
    if prompt is None:
        # If build fails, use fallback: provide all available tables
        all_available_tables = get_all_tables_from_databases(schemas, database_dir)
        if not any(all_available_tables.values()):
            return None, False, "No tables available in databases"
        # Rebuild prompt using all available tables
        prompt, _, _ = construct_cross_database_prompt(
            sql_skeleton, schemas, table_database_mapping, graph_data, sql_analysis, database_dir
        )

    # Call LLM to generate SQL
    client = get_client()
    API_CONFIG = load_config()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=API_CONFIG.get("model", "gpt-4o"),
                messages=[
                    {"role": "system", "content": "你是一个SQL专家，擅长生成跨数据库SQL查询。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=API_CONFIG.get("temperature", 0.1),
                max_tokens=API_CONFIG.get("max_tokens", 8000)
            )

            generated_sql = response.choices[0].message.content.strip()

            # Clean up SQL (remove code block markers, etc.)
            generated_sql = re.sub(r'^```sql\s*', '', generated_sql, flags=re.IGNORECASE)
            generated_sql = re.sub(r'^```\s*', '', generated_sql)
            generated_sql = re.sub(r'```\s*$', '', generated_sql)
            generated_sql = generated_sql.strip()

            # Validate SQL syntax
            try:
                sqlparse.parse(generated_sql)
            except:
                if attempt < max_retries - 1:
                    continue
                return idx if match else hash_id, False, "SQL syntax error"

            # Execute SQL and get results
            # Use ATTACH DATABASE to execute cross-database SQL
            results = None
            execution_error = None

            try:
                # Method 1: use ATTACH DATABASE for true cross-database query
                results, success = execute_cross_database_sql_with_attach(
                    generated_sql, databases, database_dir, table_database_mapping
                )

                if not success:
                    # Method 2: if ATTACH fails, try single-database format (fallback)
                    single_db_sql = convert_to_single_database_sql(generated_sql, table_database_mapping)

                    # Try executing on involved databases (prefer first database)
                    for db_name in databases:
                        db_path = os.path.join(database_dir, db_name, f"{db_name}.db")
                        if os.path.exists(db_path):
                            results, success = execute_sql_on_database(single_db_sql, db_path)
                            if success:
                                break  # Executed successfully, exit loop

                    # If all methods fail, record error
                    if not success:
                        execution_error = "Cannot execute SQL on any database (both ATTACH and single-database format failed)"
                        results = None
                else:
                    # ATTACH succeeded; results may be an empty list (empty results also count as success)
                    if results is None:
                        results = []
            except Exception as e:
                execution_error = f"Execution exception: {str(e)}"

            # Save results (limit result count, consistent with single-database)
            saved_results = []
            if results is not None:
                # Save only first 10 results (consistent with single-database)
                saved_results = results[:10] if len(results) > 10 else results
                # Convert to list format (ensure JSON serializable)
                saved_results = [list(row) for row in saved_results]

            # Save results
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
                    'model': API_CONFIG.get("model", "gpt-4o"),
                    'attempt': attempt + 1
                }
            }

            # If execution error occurred, record it in metadata
            if execution_error:
                result['metadata']['execution_error'] = execution_error

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            return idx if match else hash_id, True, "Success"

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return idx if match else hash_id, False, f"Generation failed: {str(e)}"

    return idx if match else hash_id, False, "Maximum retries reached"

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
                       default='benchmark/data/beijing/output/cross_db_single',
                       help='Output directory')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum number of retries')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='Number of concurrent threads')

    args = parser.parse_args()

    # Load cross-database SQL skeletons
    print(f"Loading cross-database SQL skeleton: {args.skeleton_file}")
    with open(args.skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)

    print(f"Total {len(skeletons)} SQL skeletons")

    # Get all involved databases
    all_databases = set()
    for skeleton in skeletons:
        all_databases.update(skeleton.get('databases', []))

    print(f"Involved databases: {sorted(all_databases)}")

    # Load schemas for all databases
    print("\nLoading database schemas...")
    schemas = load_multiple_schemas(all_databases, args.database_dir)
    print(f"Successfully loaded {len(schemas)} database schemas")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each SQL skeleton
    print(f"\nFilling SQL skeletons...")
    success_count = 0
    failed_count = 0

    # Use thread pool for concurrent processing
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
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fill progress"):
            idx, success, message = future.result()
            if success:
                success_count += 1
            else:
                failed_count += 1
                if failed_count <= 10:  # Show only first 10 errors
                    print(f"\nFailed (idx={idx}): {message}")

    print(f"\nComplete!")
    print(f"Success: {success_count}/{len(skeletons)}")
    print(f"Failed: {failed_count}/{len(skeletons)}")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()

