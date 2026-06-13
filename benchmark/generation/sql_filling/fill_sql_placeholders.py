#!/usr/bin/env python3
"""
Improved SQL skeleton filling script.

Key improvements:
1. Actually use graph structure to select relevant tables and columns
2. Use foreign key relations to select tables that can be JOINed
3. Enhanced prompts with table descriptions, column info, and foreign key relations
4. Intelligent reasoning: select the most suitable tables based on SQL skeleton semantics
5. Integrated API configuration
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

# Import graph information extraction module
try:
    from .graph_extractor import extract_relevant_nodes_from_graph, format_extracted_info_for_prompt
except ImportError:
    # If relative import fails, try absolute import
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from graph_extractor import extract_relevant_nodes_from_graph, format_extracted_info_for_prompt

# Load API configuration
def load_config(config_file=None):
    """Load configuration file"""
    try:
        from taco.core.config import load_llm_config
        return load_llm_config(config_file)
    except ImportError:
        pass

    if config_file is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_file = os.path.join(script_dir, 'config.yaml')

    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('llm', {})

    return {
        "api_url": os.environ.get("TACO_API_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("TACO_API_KEY", "your-api-key-here"),
        "model": os.environ.get("TACO_MODEL", "gpt-4o-mini"),
        "temperature": 0.1,
        "max_tokens": 8000,
    }

# Global configuration and client (lazy initialization)
API_CONFIG = None
client = None

def get_client():
    """Get OpenAI client (lazy initialization)"""
    global client, API_CONFIG
    if client is None:
        if API_CONFIG is None:
            API_CONFIG = load_config()
        # Ensure base_url format is correct (no trailing slash; OpenAI SDK adds it automatically)
        api_url = API_CONFIG["api_url"].rstrip('/')
        client = OpenAI(
            base_url=api_url,
            api_key=API_CONFIG["api_key"]
        )
    return client

def load_schema(schema_file):
    """Load database schema information"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if standard schema format (contains 'tables' key)
    if 'tables' in data:
        return data
    
    # If not standard format, extract schema from database JSON file
    # Database JSON format: {table_name: {columns: [...], data: [...]}}
    schema = {'tables': []}
    for table_name, table_data in data.items():
        columns = []
        if 'columns' in table_data:
            # Extract column names from columns list
            for col_name in table_data['columns']:
                # Infer data type (default TEXT)
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'  # Default type
                })
        
        schema['tables'].append({
            'table_name': table_name,
            'table_comment': table_name,
            'table_description': 'No description available.',
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        })
    
    return schema

def load_graph(graph_file):
    """Load graph file"""
    if not os.path.exists(graph_file):
        return None
    return nx.read_graphml(graph_file)

def load_graph_metadata(metadata_file):
    """Load graph metadata"""
    if not os.path.exists(metadata_file):
        return None
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_sql_skeleton(sql_skeleton):
    """
    Analyze SQL skeleton and extract semantic information.
    Returns:
    - has_join: whether JOIN is present
    - has_subquery: whether subquery is present
    - has_aggregate: whether aggregate function is present
    - required_tables: estimated number of tables needed
    """
    sql_upper = sql_skeleton.upper()
    
    has_join = 'JOIN' in sql_upper
    has_subquery = '(' in sql_skeleton and 'SELECT' in sql_upper
    has_aggregate = any(func in sql_upper for func in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP BY'])
    
    # Estimate required table count from placeholder count and JOIN count
    num_placeholders = sql_skeleton.count('_')
    num_joins = sql_upper.count('JOIN')
    
    if has_join:
        required_tables = min(num_joins + 1, 3)  # JOIN typically requires 2-3 tables
    elif num_placeholders <= 3:
        required_tables = 1
    else:
        required_tables = min(2, num_placeholders // 3)
    
    return {
        'has_join': has_join,
        'has_subquery': has_subquery,
        'has_aggregate': has_aggregate,
        'required_tables': required_tables,
        'num_joins': num_joins
    }

def find_tables_with_common_columns(schema_info, min_common_cols=1):
    """
    Find table pairs with common columns that can be used for JOIN.
    
    Returns: [(table1, table2, common_columns), ...]
    """
    table_column_sets = {}
    for table_name, columns in schema_info['columns'].items():
        # Extract column names (strip table name prefix)
        column_names = set()
        for col in columns:
            if '.' in col:
                column_names.add(col.split('.', 1)[1])
            else:
                column_names.add(col)
        table_column_sets[table_name] = column_names
    
    common_column_pairs = []
    table_list = list(table_column_sets.keys())
    
    for i, table1 in enumerate(table_list):
        for table2 in table_list[i+1:]:
            common_cols = table_column_sets[table1] & table_column_sets[table2]
            if len(common_cols) >= min_common_cols:
                common_column_pairs.append((table1, table2, list(common_cols)))
    
    return common_column_pairs

def find_common_columns_for_tables(selected_tables, selected_columns):
    """
    Find common columns among selected tables for JOIN hints.
    
    Returns: [(table1, table2, common_columns), ...]
    """
    # Extract column names per table (strip table name prefix)
    table_column_sets = {}
    for table in selected_tables:
        if table in selected_columns:
            column_names = set()
            for col in selected_columns[table]:
                if '.' in col:
                    column_names.add(col.split('.', 1)[1])
                else:
                    column_names.add(col)
            table_column_sets[table] = column_names
    
    common_column_pairs = []
    for i, table1 in enumerate(selected_tables):
        for table2 in selected_tables[i+1:]:
            if table1 in table_column_sets and table2 in table_column_sets:
                common_cols = table_column_sets[table1] & table_column_sets[table2]
                if common_cols:
                    common_column_pairs.append((table1, table2, list(common_cols)))
    
    return common_column_pairs

def select_tables_using_graph(G, metadata, sql_analysis, schema_info):
    """
    Intelligently select tables using metadata (graph structure no longer required; G kept for backward compatibility).
    
    Strategy:
    1. If JOIN is present, prefer table pairs with foreign key relations
    2. If JOIN is needed but no foreign keys exist, find table pairs with common columns
    3. If neither applies, randomly select tables (preserves original logic)
    4. Consider column count per table and select tables with moderate column counts
    """
    if metadata is None:
        # Fall back to random selection if no metadata
        return select_random_tables(schema_info, sql_analysis['required_tables'])
    
    all_tables = list(metadata['table_info'].keys())
    required_tables = sql_analysis['required_tables']
    
    if required_tables == 1:
        # Single-table query, random selection
        selected_table = random.choice(all_tables)
        return [selected_table], {selected_table: metadata['table_info'][selected_table]['columns']}
    
    # Multi-table query, prefer tables with foreign key relations
    fk_relations = metadata['foreign_key_relations']
    
    if sql_analysis['has_join']:
        # JOIN operation required
        if fk_relations:
            # Foreign key relations exist, prefer using them
            fk_relation = random.choice(fk_relations)
            source_table = fk_relation['source_table']
            target_table = fk_relation['target_table']
            
            selected_tables = [source_table, target_table]
            
            # Add more tables randomly if needed
            if required_tables > 2:
                remaining_tables = [t for t in all_tables if t not in selected_tables]
                if remaining_tables:
                    additional = random.sample(remaining_tables, min(required_tables - 2, len(remaining_tables)))
                    selected_tables.extend(additional)
        else:
            # No foreign key relations, find table pairs with common columns
            common_column_pairs = find_tables_with_common_columns(schema_info, min_common_cols=1)
            
            if common_column_pairs:
                # Randomly select a table pair with common columns
                table1, table2, common_cols = random.choice(common_column_pairs)
                selected_tables = [table1, table2]
                
                # Add more tables randomly if needed
                if required_tables > 2:
                    remaining_tables = [t for t in all_tables if t not in selected_tables]
                    if remaining_tables:
                        additional = random.sample(remaining_tables, min(required_tables - 2, len(remaining_tables)))
                        selected_tables.extend(additional)
            else:
                # No common columns, random selection (JOIN may fail, but SQL can still be generated)
                selected_tables = random.sample(all_tables, min(required_tables, len(all_tables)))
    else:
        # No JOIN needed, random selection
        selected_tables = random.sample(all_tables, min(required_tables, len(all_tables)))
    
    # Build selected table and column information
    selected_columns = {}
    for table in selected_tables:
        if table in metadata['table_info']:
            selected_columns[table] = metadata['table_info'][table]['columns']
        else:
            # Fallback: get from schema_info
            selected_columns[table] = schema_info['columns'].get(table, [])
    
    return selected_tables, selected_columns

def select_random_tables(schema_info, num_tables=2):
    """Randomly select tables (preserves original logic)"""
    all_tables = schema_info['tables']
    selected_tables = random.sample(all_tables, min(num_tables, len(all_tables)))
    selected_columns = {}
    for table in selected_tables:
        selected_columns[table] = schema_info['columns'].get(table, [])
    return selected_tables, selected_columns

def extract_schema_info(schema):
    """Extract table names and column names from schema"""
    schema_info = {
        'tables': [],
        'columns': {}
    }
    for table in schema['tables']:
        table_name = table['table_name']
        schema_info['tables'].append(table_name)
        columns = []
        for column in table['columns']:
            column_name = column['column_name']
            full_column_name = f"{table_name}.{column_name}"
            columns.append(full_column_name)
        schema_info['columns'][table_name] = columns
    return schema_info

def extract_graph_metadata_from_loaded(metadata_dict):
    """
    Extract information from loaded metadata dictionary (using original names).
    Supports two formats:
    1. Legacy format: contains node_id_map, table_info, column_info
    2. New format (optimized): directly contains tables and foreign_keys
    """
    # Check if new format (optimized version)
    if 'tables' in metadata_dict and isinstance(metadata_dict['tables'], dict):
        # New format: use tables and foreign_keys directly
        table_info = {}
        column_info = {}
        foreign_key_relations = []
        
        # Process table information
        for table_name, table_data in metadata_dict['tables'].items():
            table_info[table_name] = {
                'name': table_data.get('name', table_name),
                'comment': table_data.get('comment', ''),
                'description': table_data.get('description', 'No description available.'),
                'columns': []
            }
            
            # Process column information
            for col in table_data.get('columns', []):
                col_name = col.get('name', '')
                full_column_name = f"{table_name}.{col_name}"
                column_info[full_column_name] = {
                    'full_name': full_column_name,
                    'table': table_name,
                    'column': col_name,
                    'data_type': col.get('data_type', 'TEXT')
                }
                table_info[table_name]['columns'].append(full_column_name)
        
        # Process foreign key relations
        for fk in metadata_dict.get('foreign_keys', []):
            source_table = fk.get('source_table', '')
            source_column = fk.get('source_column', '')
            target_table = fk.get('target_table', '')
            target_column = fk.get('target_column', '')
            
            if source_table and source_column and target_table and target_column:
                source_full = f"{source_table}.{source_column}"
                target_full = f"{target_table}.{target_column}"
                foreign_key_relations.append({
                    'source': source_full,
                    'target': target_full,
                    'source_table': source_table,
                    'target_table': target_table
                })
        
        return {
            'foreign_key_relations': foreign_key_relations,
            'table_info': table_info,
            'column_info': column_info
        }
    
    # Legacy format: restore original names using node_id_map
    foreign_key_relations = metadata_dict.get('foreign_key_relations', [])
    table_info = {}
    column_info = {}
    node_id_map = metadata_dict.get('node_id_map', {})
    
    # Process table information (using original names)
    for cleaned_id, original_name in node_id_map.items():
        if cleaned_id in metadata_dict.get('table_info', {}):
            table_meta = metadata_dict['table_info'][cleaned_id]
            table_info[original_name] = {
                'name': original_name,
                'comment': table_meta.get('comment', ''),
                'description': table_meta.get('description', 'No description available.'),
                'columns': []
            }
    
    # Process column information
    for cleaned_id, original_name in node_id_map.items():
        if cleaned_id in metadata_dict.get('column_info', {}):
            col_meta = metadata_dict['column_info'][cleaned_id]
            if '.' in original_name:
                table_name = original_name.split('.')[0]
            else:
                table_name = col_meta.get('table', '')
            
            column_info[original_name] = {
                'full_name': original_name,
                'table': table_name,
                'column': col_meta.get('column', ''),
                'data_type': col_meta.get('data_type', 'TEXT')
            }
            if table_name in table_info:
                table_info[table_name]['columns'].append(original_name)
    
    # Process foreign key relations (using original names)
    fk_relations_original = []
    for fk in foreign_key_relations:
        source_cleaned = fk.get('source', '')
        target_cleaned = fk.get('target', '')
        source_original = node_id_map.get(source_cleaned, source_cleaned)
        target_original = node_id_map.get(target_cleaned, target_cleaned)
        fk_relations_original.append({
            'source': source_original,
            'target': target_original,
            'source_table': fk.get('source_table', ''),
            'target_table': fk.get('target_table', '')
        })
    
    return {
        'foreign_key_relations': fk_relations_original,
        'table_info': table_info,
        'column_info': column_info
    }

def format_foreign_key_relations(metadata, selected_tables):
    """Format foreign key relation information for prompt"""
    if not metadata or 'foreign_key_relations' not in metadata:
        return ""
    
    fk_relations = metadata['foreign_key_relations']
    relevant_fks = [
        fk for fk in fk_relations 
        if fk['source_table'] in selected_tables and fk['target_table'] in selected_tables
    ]
    
    if not relevant_fks:
        return ""
    
    fk_text = "\nForeign Key Relations:\n"
    for fk in relevant_fks:
        fk_text += f"- {fk['source']} references {fk['target']}\n"
        fk_text += f"  (Table {fk['source_table']} can JOIN with table {fk['target_table']} via {fk['source'].split('.')[1]} and {fk['target'].split('.')[1]})\n"
    
    return fk_text

def format_table_info(metadata, selected_tables, schema):
    """Format table information for prompt"""
    table_info_text = "\nTable Details:\n"
    
    for table_name in selected_tables:
        table_info_text += f"\nTable: {table_name}\n"
        
        # Get table description from metadata
        if metadata and table_name in metadata.get('table_info', {}):
            table_meta = metadata['table_info'][table_name]
            if table_meta.get('description') and table_meta['description'] != 'No description available.':
                table_info_text += f"Description: {table_meta['description']}\n"
            if table_meta.get('comment'):
                table_info_text += f"Comment: {table_meta['comment']}\n"
        
        # Get column information from schema
        table_info_text += "Columns:\n"
        for table in schema['tables']:
            if table['table_name'] == table_name:
                for column in table['columns']:
                    column_name = column['column_name']
                    data_type = column.get('data_type', 'TEXT')
                    full_column_name = f"{table_name}.{column_name}"
                    table_info_text += f"  - {full_column_name} (Type: {data_type})\n"
                break
    
    return table_info_text

def construct_enhanced_prompt(sql_skeleton, selected_tables, selected_columns, 
                              metadata, schema, sql_analysis, cross_database=False):
    """
    Build enhanced prompt containing:
    1. SQL skeleton
    2. Table details (description, comment)
    3. Column information (data types)
    4. Foreign key relations
    5. SQL skeleton analysis results
    """
    def quote_identifier(identifier):
        """Wrap identifier in double quotes so SQLite handles Chinese and special characters correctly"""
        # Escape double quotes
        escaped = str(identifier).replace('"', '""')
        return f'"{escaped}"'
    
    # Format table names (wrap all table names in double quotes)
    tables = ', '.join([quote_identifier(table) for table in selected_tables])
    
    # Format column names (wrap all column names in double quotes)
    columns = []
    for table in selected_tables:
        if table in selected_columns:
            # Column format: table.column, wrap each part separately
            for col in selected_columns[table]:
                if '.' in col:
                    # If already in "table.column" format, process each part separately
                    parts = col.split('.', 1)
                    if len(parts) == 2:
                        table_part, col_part = parts
                        quoted_col = f'{quote_identifier(table_part)}.{quote_identifier(col_part)}'
                    else:
                        quoted_col = quote_identifier(col)
                else:
                    quoted_col = quote_identifier(col)
                columns.append(quoted_col)
    columns_str = ', '.join(columns)
    
    # Format table details
    table_info_text = format_table_info(metadata, selected_tables, schema)
    
    # Format foreign key relations
    fk_text = format_foreign_key_relations(metadata, selected_tables)
    
    # SQL skeleton analysis hints
    analysis_hints = ""
    if sql_analysis['has_join']:
        analysis_hints += "\nHint: This SQL skeleton contains JOIN operations.\n"
        if fk_text:
            analysis_hints += "  - Prefer using foreign key relations to join tables (see Foreign Key Relations below).\n"
        # Check if common columns are available for JOIN
        if len(selected_tables) >= 2:
            common_cols_info = find_common_columns_for_tables(selected_tables, selected_columns)
            if common_cols_info:
                analysis_hints += "  - If no foreign key relations exist, common columns can be used for JOIN.\n"
                analysis_hints += "  - The following table pairs have common columns usable for JOIN conditions:\n"
                for table1, table2, common_cols in common_cols_info:
                    common_cols_str = ', '.join([f'"{col}"' for col in common_cols[:3]])  # Show first 3 only
                    analysis_hints += f"    * {table1[:50]}... and {table2[:50]}... common columns: {common_cols_str}\n"
    if sql_analysis['has_aggregate']:
        analysis_hints += "Hint: This SQL skeleton contains aggregate functions, ensure GROUP BY clause is correct.\n"
    if sql_analysis['has_subquery']:
        analysis_hints += "Hint: This SQL skeleton contains subqueries, ensure subquery syntax is correct.\n"
    
    if cross_database:
        databases = ', '.join(list(set([table.split('.')[0] for table in selected_tables if '.' in table])))
        prompt = f"""Please fill in the placeholders "_" in the following SQL skeleton with actual table names and column names to generate a complete and executable SQL statement for SQLite.

Strict Requirements:
- **Output only the final complete SQL statement, do not repeat the prompt content.**
- **The generated SQL must be syntactically correct and can be directly executed on SQLite to get results.**
- **Do not add any additional explanations, comments, or output formatting (code blocks, spaces, etc.).**
- **Table names, column names, WHERE conditions, etc. must be from the given tables and columns.**
- **All table names and column names must be wrapped in double quotes (including Chinese and special characters), for example: "table_name" or "table_name"."column_name"**
- **SQLite supports Chinese table and column names; just wrap them correctly with double quotes.**
- **You can adjust the given SQL skeleton to generate a more reasonable SQL statement.**
- **If the SQL skeleton contains JOIN, please prioritize using foreign key relations to join tables. If there are no foreign key relations, you can use common columns for JOIN (see hints below).**

SQL Skeleton:
{sql_skeleton}

Available Databases:
{databases}

Available Table Names:
{tables}

Available Column Names (format: table_name.column_name):
{columns_str}
{table_info_text}
{fk_text}
{analysis_hints}

Please output only the generated complete SQL statement:
"""
    else:
        prompt = f"""Please fill in the placeholders "_" in the following SQL skeleton with actual table names and column names to generate a complete and executable SQL statement for SQLite.

Strict Requirements:
- **Output only the final complete SQL statement, do not repeat the prompt content.**
- **The generated SQL must be syntactically correct and can be directly executed on SQLite to get results.**
- **Do not add any additional explanations, comments, or output formatting (code blocks, spaces, etc.).**
- **Table names, column names, WHERE conditions, etc. must be from the given tables and columns.**
- **All table names and column names must be wrapped in double quotes (including Chinese and special characters), for example: "table_name" or "table_name"."column_name"**
- **SQLite supports Chinese table and column names; just wrap them correctly with double quotes.**
- **You can adjust the given SQL skeleton to generate a more reasonable SQL statement.**
- **If the SQL skeleton contains JOIN, please prioritize using foreign key relations to join tables. If there are no foreign key relations, you can use common columns for JOIN (see hints below).**

SQL Skeleton:
{sql_skeleton}

Available Table Names:
{tables}

Available Column Names (format: table_name.column_name):
{columns_str}
{table_info_text}
{fk_text}
{analysis_hints}

Please output only the generated complete SQL statement:
"""
    
    return prompt.strip()

def construct_compact_prompt(sql_framework, extracted_info, schema):
    """
    Build prompt from compact information extracted from graph file.
    This version includes only tables and columns related to the SQL skeleton, significantly reducing prompt size.
    """
    def quote_identifier(identifier):
        """Wrap identifier in double quotes so SQLite handles Chinese and special characters correctly"""
        escaped = str(identifier).replace('"', '""')
        return f'"{escaped}"'
    
    tables = extracted_info.get('tables', [])
    table_info = extracted_info.get('table_info', {})
    columns = extracted_info.get('columns', {})
    column_info = extracted_info.get('column_info', {})
    foreign_keys = extracted_info.get('foreign_keys', [])
    sql_analysis = extracted_info.get('sql_analysis', {})
    
    # Format table names
    tables_str = ', '.join([quote_identifier(table) for table in tables])
    
    # Format column names
    columns_list = []
    for table in tables:
        if table in columns:
            for col_name in columns[table]:
                if '.' in col_name:
                    parts = col_name.split('.', 1)
                    if len(parts) == 2:
                        table_part, col_part = parts
                        quoted_col = f'{quote_identifier(table_part)}.{quote_identifier(col_part)}'
                    else:
                        quoted_col = quote_identifier(col_name)
                else:
                    quoted_col = f'{quote_identifier(table)}.{quote_identifier(col_name)}'
                columns_list.append(quoted_col)
    columns_str = ', '.join(columns_list)
    
    # Format table details
    table_info_text = "\nTable Details:\n"
    for table_name in tables:
        table_info_text += f"\nTable: {table_name}\n"
        
        if table_name in table_info:
            info = table_info[table_name]
            if info.get('description') and info['description'] != 'No description available.':
                table_info_text += f"Description: {info['description']}\n"
            if info.get('comment'):
                table_info_text += f"Comment: {info['comment']}\n"
        
        # Column information
        if table_name in columns:
            table_info_text += "Columns:\n"
            for col_name in columns[table_name]:
                if col_name in column_info:
                    col_info = column_info[col_name]
                    data_type = col_info.get('data_type', 'TEXT')
                    table_info_text += f"  - {col_name} (Type: {data_type})\n"
    
    # Format foreign key relations
    fk_text = ""
    if foreign_keys:
        fk_text = "\nForeign Key Relations (usable for JOIN):\n"
        for fk in foreign_keys:
            source_full = f"{fk['source_table']}.{fk['source_column']}"
            target_full = f"{fk['target_table']}.{fk['target_column']}"
            fk_text += f"- {source_full} references {target_full}\n"
            fk_text += f"  (Table {fk['source_table']} can JOIN with table {fk['target_table']} via {fk['source_column']} and {fk['target_column']})\n"
    
    # SQL skeleton analysis hints
    analysis_hints = ""
    if sql_analysis.get('has_join'):
        analysis_hints += "\nHint: This SQL skeleton contains JOIN operations, use foreign key relations to join tables.\n"
    if sql_analysis.get('has_aggregate'):
        analysis_hints += "Hint: This SQL skeleton contains aggregate functions, ensure GROUP BY clause is correct.\n"
    if sql_analysis.get('has_subquery'):
        analysis_hints += "Hint: This SQL skeleton contains subqueries, ensure subquery syntax is correct.\n"
    
    prompt = f"""Please fill in the placeholders "_" in the following SQL skeleton with actual table names and column names to generate a complete and executable SQL statement for SQLite.

Strict Requirements:
- **Output only the final complete SQL statement, do not repeat the prompt content.**
- **The generated SQL must be syntactically correct and can be directly executed on SQLite to get results.**
- **Do not add any additional explanations, comments, or output formatting (code blocks, spaces, etc.).**
- **Table names, column names, WHERE conditions, etc. must be from the given tables and columns.**
- **All table names and column names must be wrapped in double quotes (including Chinese and special characters), for example: "table_name" or "table_name"."column_name"**
- **SQLite supports Chinese table and column names; just wrap them correctly with double quotes.**
- **You can adjust the given SQL skeleton to generate a more reasonable SQL statement.**
- **If the SQL skeleton contains JOIN, please prioritize using foreign key relations to join tables. If there are no foreign key relations, you can use common columns for JOIN (see hints below).**

SQL Skeleton:
{sql_framework}

Available Table Names (filtered by SQL skeleton, {len(tables)} total):
{tables_str}

Available Column Names (format: table_name.column_name, filtered by SQL skeleton):
{columns_str}
{table_info_text}
{fk_text}
{analysis_hints}

Please output only the generated complete SQL statement:
"""
    
    return prompt.strip()

def generate_text(prompt):
    """Call LLM to generate SQL"""
    try:
        client = get_client()
        response = client.chat.completions.create(
            model=API_CONFIG["model"],
            temperature=API_CONFIG["temperature"],
            max_tokens=API_CONFIG["max_tokens"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in SQL generation."},
                {"role": "user", "content": prompt},
            ],
        )
        assistant_reply = response.choices[0].message.content
        return assistant_reply.strip()
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return None

def extract_sql_from_response(response):
    """Extract SQL statement from model output"""
    sql_statement = response.strip()
    # Remove possible code block markers
    sql_statement = re.sub(r'```sql\s*', '', sql_statement, flags=re.IGNORECASE)
    sql_statement = re.sub(r'```\s*', '', sql_statement)
    # Ensure it starts with "SELECT"
    if not sql_statement.upper().startswith('SELECT'):
        match = re.search(r'(SELECT\s.*)', sql_statement, re.IGNORECASE | re.DOTALL)
        if match:
            sql_statement = match.group(1).strip()
    return sql_statement

def is_valid_sql(sql_statement):
    """Validate SQL statement syntax"""
    try:
        parsed = sqlparse.parse(sql_statement)
        if parsed and len(parsed) > 0:
            return True
        else:
            return False
    except Exception:
        return False

def execute_single_db_sql(sql, db_path):
    """Execute SQL statement on a single database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        if results:
            return results, True
        else:
            return [], False  # Empty result still counts as success
    except sqlite3.Error as e:
        # Record more detailed error information
        error_msg = str(e)
        # Do not print every error to avoid excessive output
        return None, False
    except Exception as e:
        # Record other types of errors
        error_msg = str(e)
        return None, False

def process_single_sql_skeleton(args):
    """Process a single SQL skeleton (for concurrent processing)"""
    idx, sql_skeleton, database_name, schema, schema_info, graph_dir, single_output_path, schema_file, max_retries = args
    
    # Extract SQL skeleton string
    if isinstance(sql_skeleton, dict):
        sql_framework = sql_skeleton.get('sql_framework', '')
    else:
        sql_framework = sql_skeleton
    
    if not sql_framework:
        return idx, False, "SQL skeleton is empty"
    
    # Check if output file already exists
    output_file = os.path.join(single_output_path, f'generated_sql_{idx}.json')
    if os.path.exists(output_file):
        return idx, True, "Already exists"
    
    # Analyze SQL skeleton
    sql_analysis = analyze_sql_skeleton(sql_framework)
    
    # Load metadata (prefer metadata file)
    metadata_file = os.path.join(graph_dir, database_name, f"{database_name}_metadata_{idx}.json")
    metadata_dict = load_graph_metadata(metadata_file)
    
    # Extract information from metadata (using original names)
    if metadata_dict:
        metadata = extract_graph_metadata_from_loaded(metadata_dict)
    else:
        metadata = None
    
    # Try to extract key information from graph file (prefer graph extraction for better accuracy)
    graph_file = os.path.join(graph_dir, database_name, f"{database_name}_graph_{idx}.graphml")
    extracted_info = None
    use_extracted_info = False
    
    if os.path.exists(graph_file):
        try:
            # Extract key information related to SQL skeleton from graph file
            G = load_graph(graph_file)
            extracted_info = extract_relevant_nodes_from_graph(G, sql_framework, max_tables=5, max_columns_per_table=10)
            
            if extracted_info and len(extracted_info.get('tables', [])) > 0:
                use_extracted_info = True
                # Convert extracted information to metadata format
                metadata = {
                    'foreign_key_relations': [
                        {
                            'source': f"{fk['source_table']}.{fk['source_column']}",
                            'target': f"{fk['target_table']}.{fk['target_column']}",
                            'source_table': fk['source_table'],
                            'target_table': fk['target_table']
                        }
                        for fk in extracted_info.get('foreign_keys', [])
                    ],
                    'table_info': extracted_info.get('table_info', {}),
                    'column_info': extracted_info.get('column_info', {})
                }
                
                # Use extracted tables and columns
                selected_tables = extracted_info.get('tables', [])
                selected_columns = extracted_info.get('columns', {})
                
                # Supplement with random selection if not enough tables extracted
                if len(selected_tables) < sql_analysis['required_tables']:
                    remaining_tables = [t for t in schema_info['tables'] if t not in selected_tables]
                    if remaining_tables:
                        additional = random.sample(
                            remaining_tables, 
                            min(sql_analysis['required_tables'] - len(selected_tables), len(remaining_tables))
                        )
                        selected_tables.extend(additional)
                        for table in additional:
                            if table in schema_info['columns']:
                                selected_columns[table] = schema_info['columns'][table]
        except Exception as e:
            # Graph file load failed, fall back to metadata or random selection
            use_extracted_info = False
    
    # If no information extracted from graph file, use metadata or random selection
    if not use_extracted_info:
        # Load metadata (prefer metadata file)
        if not metadata_dict:
            metadata_dict = load_graph_metadata(metadata_file)
        
        # Extract information from metadata (using original names)
        if metadata_dict:
            metadata = extract_graph_metadata_from_loaded(metadata_dict)
        else:
            metadata = None
        
        # Select tables using metadata (graph structure no longer required)
        selected_tables, selected_columns = select_tables_using_graph(
            None, metadata, sql_analysis, schema_info
        )
    
    # Build prompt (use compact prompt if information was extracted from graph file)
    if use_extracted_info and extracted_info:
        prompt = construct_compact_prompt(sql_framework, extracted_info, schema)
    else:
        prompt = construct_enhanced_prompt(
            sql_framework, selected_tables, selected_columns,
            metadata, schema, sql_analysis, cross_database=False
        )
    
    # Try to generate SQL (with retries)
    sql_statement = None
    error_info = None
    for attempt in range(1, max_retries + 1):
        try:
            sql_statement = generate_text(prompt)
            if not sql_statement:
                error_info = "LLM generation failed"
                if attempt < max_retries:
                    time.sleep(1)  # Wait before retry
                continue
            
            sql_statement = extract_sql_from_response(sql_statement)
            
            if is_valid_sql(sql_statement) and sql_statement.upper().startswith('SELECT'):
                # Get database path
                db_path = schema_file.replace('.json', '.db')
                if not os.path.exists(db_path):
                    error_info = f"Database file does not exist: {db_path}"
                    break
                
                # Execute SQL
                results, success = execute_single_db_sql(sql_statement, db_path)
                if success:
                    # Save results
                    save_data = {
                        'sql': sql_statement,
                        'results': results[:10] if results else [],  # Save only first 10 results
                        'sql_skeleton': sql_framework,
                        'database': database_name,
                        'tables': {table: selected_columns[table] for table in selected_tables},
                        'metadata': {
                            'has_join': sql_analysis['has_join'],
                            'has_subquery': sql_analysis['has_subquery'],
                            'has_aggregate': sql_analysis['has_aggregate']
                        }
                    }
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    
                    return idx, True, "Success"
                else:
                    error_info = "SQL execution failed"
                    if attempt < max_retries:
                        time.sleep(1)  # Wait before retry
            else:
                error_info = "SQL syntax validation failed"
                if attempt < max_retries:
                    time.sleep(1)  # Wait before retry
        except Exception as e:
            error_info = f"Processing exception: {str(e)}"
            if attempt < max_retries:
                time.sleep(1)  # Wait before retry
    
    # Save failure information
    if not sql_statement or not is_valid_sql(sql_statement):
        error_file = output_file.replace('.json', '_error.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump({
                'sql_skeleton': sql_framework,
                'database': database_name,
                'error': error_info,
                'generated_sql': sql_statement if sql_statement else None
            }, f, ensure_ascii=False, indent=2)
        return idx, False, error_info
    
    return idx, True, "Success"

def process_single_database(database_name, skeleton_file, schema_file, graph_dir, output_dir, max_retries=3, max_workers=None):
    """Process SQL skeleton filling for a single database (supports concurrency)"""
    # Load schema
    schema = load_schema(schema_file)
    schema_info = extract_schema_info(schema)
    
    # Load SQL skeletons
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        sql_skeletons = json.load(f)
    
    # Create output directory
    single_output_path = os.path.join(output_dir, 'single', database_name)
    os.makedirs(single_output_path, exist_ok=True)
    
    # Get concurrency count
    if max_workers is None:
        max_workers = API_CONFIG.get('max_workers', 20) if API_CONFIG else 20
    
    print(f"Processing database '{database_name}', {len(sql_skeletons)} SQL skeletons...")
    print(f"Concurrency: {max_workers}, max retries: {max_retries}")
    
    success_count = 0
    fail_count = 0
    
    # Prepare task arguments
    tasks = []
    for idx, sql_skeleton in enumerate(sql_skeletons):
        tasks.append((
            idx, sql_skeleton, database_name, schema, schema_info, 
            graph_dir, single_output_path, schema_file, max_retries
        ))
    
    # Process concurrently with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_single_sql_skeleton, task): task[0] for task in tasks}
        
        # Show progress with tqdm
        with tqdm(total=len(tasks), desc=f"{database_name} progress") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_idx, success, message = future.result()
                    if success:
                        if message != "Already exists":
                            success_count += 1
                    else:
                        fail_count += 1
                except Exception as e:
                    fail_count += 1
                    print(f"Exception processing index {idx}: {e}")
                finally:
                    pbar.update(1)
    
    print(f"Database '{database_name}' complete: success {success_count}, failed {fail_count}")
    return success_count, fail_count

def main():
    global API_CONFIG
    
    parser = argparse.ArgumentParser(description='Fill SQL skeleton placeholders (improved version)')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser.add_argument('--skeleton_dir', type=str, default=None,
                       help='SQL skeleton directory (default: ../../data/beijing/output/sql_skeleton)')
    parser.add_argument('--database_dir', type=str, default=None,
                       help='Database directory (default: ../../data/beijing/database)')
    parser.add_argument('--graph_dir', type=str, default=None,
                       help='Graph directory (default: ../../data/beijing/output/graph)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../../data/beijing/output)')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum retry times (default: 3)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path (default: ./config.yaml)')
    
    args = parser.parse_args()
    
    # Load configuration
    API_CONFIG = load_config(args.config)
    
    # Set default paths
    if args.skeleton_dir is None:
        args.skeleton_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'sql_skeleton')
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database')
    if args.graph_dir is None:
        args.graph_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'graph')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output')
    
    # Convert to absolute paths
    args.skeleton_dir = os.path.abspath(args.skeleton_dir)
    args.database_dir = os.path.abspath(args.database_dir)
    args.graph_dir = os.path.abspath(args.graph_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Get all SQL skeleton files
    skeleton_files = [f for f in os.listdir(args.skeleton_dir) if f.endswith('_sql_skeleton.json')]
    
    print(f"Found {len(skeleton_files)} database SQL skeleton files")
    
    total_success = 0
    total_fail = 0
    
    for skeleton_file in tqdm(skeleton_files, desc="Overall progress"):
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        
        skeleton_path = os.path.join(args.skeleton_dir, skeleton_file)
        schema_path = os.path.join(args.database_dir, database_name, f"{database_name}.json")
        
        success, fail = process_single_database(
            database_name, skeleton_path, schema_path,
            args.graph_dir, args.output_dir, args.max_retries
        )
        
        total_success += success
        total_fail += fail
    
    print(f"\n{'='*60}")
    print(f"✓ All databases processed!")
    print(f"Total: success {total_success}, failed {total_fail}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

