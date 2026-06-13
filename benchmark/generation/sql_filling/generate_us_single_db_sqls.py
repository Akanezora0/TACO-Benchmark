#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-database SQL generation script for the US dataset.

Based on the beijing dataset generation pipeline, adapted for the US dataset (English).
"""

import sys
import os
from pathlib import Path

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import importlib.util
import argparse
import json
from tqdm import tqdm

# Dynamic import (module name starts with a digit)
spec = importlib.util.spec_from_file_location(
    "fill_module", 
    os.path.join(script_dir, "fill_sql_placeholders.py")
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)

# Override prompt with English version
def construct_english_prompt(sql_skeleton, selected_tables, selected_columns, 
                              metadata, schema, sql_analysis, cross_database=False):
    """
    Build English prompt (for US dataset).
    """
    def quote_identifier(identifier):
        """Wrap identifier in double quotes"""
        escaped = str(identifier).replace('"', '""')
        return f'"{escaped}"'
    
    # Format table names
    tables = ', '.join([quote_identifier(table) for table in selected_tables])
    
    # Format column names
    columns = []
    for table in selected_tables:
        if table in selected_columns:
            for col in selected_columns[table]:
                if '.' in col:
                    parts = col.split('.', 1)
                    if len(parts) == 2:
                        table_part, col_part = parts
                        quoted_col = f'{quote_identifier(table_part)}.{quote_identifier(col_part)}'
                    else:
                        quoted_col = quote_identifier(col)
                else:
                    quoted_col = f'{quote_identifier(table)}.{quote_identifier(col)}'
                columns.append(quoted_col)
    columns_str = ', '.join(columns)
    
    # Format table details
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
        
        # Get column info from schema
        table_info_text += "Columns:\n"
        for table in schema['tables']:
            if table['table_name'] == table_name:
                for column in table['columns']:
                    column_name = column['column_name']
                    data_type = column.get('data_type', 'TEXT')
                    full_column_name = f"{table_name}.{column_name}"
                    table_info_text += f"  - {full_column_name} (Type: {data_type})\n"
                break
    
    # Format foreign key relations
    fk_text = ""
    if metadata and 'foreign_key_relations' in metadata:
        fk_relations = metadata['foreign_key_relations']
        relevant_fks = [
            fk for fk in fk_relations 
            if fk['source_table'] in selected_tables and fk['target_table'] in selected_tables
        ]
        if relevant_fks:
            fk_text = "\nForeign Key Relations:\n"
            for fk in relevant_fks:
                fk_text += f"- {fk['source']} references {fk['target']}\n"
                fk_text += f"  (Table {fk['source_table']} can JOIN with table {fk['target_table']} via {fk['source'].split('.')[1]} and {fk['target'].split('.')[1]})\n"
    
    # SQL skeleton analysis hints
    analysis_hints = ""
    if sql_analysis['has_join']:
        analysis_hints += "\nHint: This SQL skeleton contains JOIN operations.\n"
        if fk_text:
            analysis_hints += "  - Prefer using foreign key relations to join tables (see Foreign Key Relations below).\n"
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
- **All table names and column names must be wrapped in double quotes, for example: "table_name" or "table_name"."column_name"**
- **SQLite supports table names and column names in English, just wrap them correctly with double quotes.**
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
- **All table names and column names must be wrapped in double quotes, for example: "table_name" or "table_name"."column_name"**
- **SQLite supports table names and column names in English, just wrap them correctly with double quotes.**
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

# Replace original module's prompt builder
fill_module.construct_enhanced_prompt = construct_english_prompt

# Also replace compact prompt
def construct_english_compact_prompt(sql_framework, extracted_info, schema):
    """Build English prompt from compact info extracted from graph file"""
    def quote_identifier(identifier):
        escaped = str(identifier).replace('"', '""')
        return f'"{escaped}"'
    
    tables = extracted_info.get('tables', [])
    table_info = extracted_info.get('table_info', {})
    columns = extracted_info.get('columns', {})
    foreign_keys = extracted_info.get('foreign_keys', [])
    sql_analysis = extracted_info.get('sql_analysis', {})
    
    tables_str = ', '.join([quote_identifier(table) for table in tables])
    
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
    
    table_info_text = "\nTable Details:\n"
    for table_name in tables:
        table_info_text += f"\nTable: {table_name}\n"
        if table_name in table_info:
            info = table_info[table_name]
            if info.get('description') and info['description'] != 'No description available.':
                table_info_text += f"Description: {info['description']}\n"
            if info.get('comment'):
                table_info_text += f"Comment: {info['comment']}\n"
        table_info_text += "Columns:\n"
        for table in schema['tables']:
            if table['table_name'] == table_name:
                for column in table['columns']:
                    column_name = column['column_name']
                    data_type = column.get('data_type', 'TEXT')
                    full_column_name = f"{table_name}.{column_name}"
                    table_info_text += f"  - {full_column_name} (Type: {data_type})\n"
                break
    
    fk_text = ""
    if foreign_keys:
        fk_text = "\nForeign Key Relations:\n"
        for fk in foreign_keys:
            fk_text += f"- {fk['source']} references {fk['target']}\n"
    
    analysis_hints = ""
    if sql_analysis.get('has_join'):
        analysis_hints += "\nHint: This SQL skeleton contains JOIN operations.\n"
    if sql_analysis.get('has_aggregate'):
        analysis_hints += "Hint: This SQL skeleton contains aggregate functions.\n"
    if sql_analysis.get('has_subquery'):
        analysis_hints += "Hint: This SQL skeleton contains subqueries.\n"
    
    prompt = f"""Please fill in the placeholders "_" in the following SQL skeleton with actual table names and column names to generate a complete and executable SQL statement for SQLite.

Strict Requirements:
- **Output only the final complete SQL statement, do not repeat the prompt content.**
- **The generated SQL must be syntactically correct and can be directly executed on SQLite to get results.**
- **Do not add any additional explanations, comments, or output formatting (code blocks, spaces, etc.).**
- **All table names and column names must be wrapped in double quotes.**

SQL Skeleton:
{sql_framework}

Available Table Names:
{tables_str}

Available Column Names (format: table_name.column_name):
{columns_str}
{table_info_text}
{fk_text}
{analysis_hints}

Please output only the generated complete SQL statement:
"""
    
    return prompt.strip()

fill_module.construct_compact_prompt = construct_english_compact_prompt

# Override system message to English
original_generate_text = fill_module.generate_text

def generate_text_english(prompt):
    """Call LLM to generate SQL (English version)"""
    try:
        client = fill_module.get_client()
        response = client.chat.completions.create(
            model=fill_module.API_CONFIG["model"],
            temperature=fill_module.API_CONFIG["temperature"],
            max_tokens=fill_module.API_CONFIG["max_tokens"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in SQL generation for English databases."},
                {"role": "user", "content": prompt},
            ],
        )
        assistant_reply = response.choices[0].message.content
        return assistant_reply.strip()
    except Exception as e:
        print(f"Error generating SQL: {e}")
        return None

fill_module.generate_text = generate_text_english

def main():
    parser = argparse.ArgumentParser(description='Generate single database SQLs for US dataset')
    parser.add_argument('--database_name', type=str, required=True,
                       help='Database name (e.g., "City of Austin - 1586")')
    parser.add_argument('--database_dir', type=str, 
                       default='../../data/us/database',
                       help='Database directory')
    parser.add_argument('--skeleton_dir', type=str,
                       default='../../data/us/output/sql_skeleton',
                       help='SQL skeleton directory')
    parser.add_argument('--graph_dir', type=str,
                       default='../../data/us/output/graph',
                       help='Graph file directory')
    parser.add_argument('--output_dir', type=str,
                       default='../../data/us/output',
                       help='Output directory')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum retry times (default: 3)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path (default: ./config.yaml)')
    parser.add_argument('--target_count', type=int, default=None,
                       help='Target number of SQLs to generate (default: generate all skeletons)')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='Maximum number of concurrent workers (default: from config.yaml or 20)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    database_dir = os.path.abspath(os.path.join(script_dir, args.database_dir))
    skeleton_dir = os.path.abspath(os.path.join(script_dir, args.skeleton_dir))
    graph_dir = os.path.abspath(os.path.join(script_dir, args.graph_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # Build file paths
    skeleton_file = os.path.join(skeleton_dir, f"{args.database_name}_sql_skeleton.json")
    # US dataset schema file may be under database dir as {database_name}/{database_name}.json or {database_name}/{database_name}.db
    schema_file = os.path.join(database_dir, args.database_name, f"{args.database_name}.json")
    if not os.path.exists(schema_file):
        # Try alternative paths
        schema_file = os.path.join(database_dir, args.database_name, "schema.json")
    
    # Check if files exist
    if not os.path.exists(skeleton_file):
        print(f"Error: SQL skeleton file does not exist: {skeleton_file}")
        return
    
    if not os.path.exists(schema_file):
        print(f"Error: Schema file does not exist: {schema_file}")
        print(f"Tried: {schema_file}")
        return
    
    # Load config
    config_file = args.config if args.config else os.path.join(script_dir, 'config.yaml')
    fill_module.API_CONFIG = fill_module.load_config(config_file)
    
    # Determine max_workers (compute early for display)
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = fill_module.API_CONFIG.get('max_workers', 20) if fill_module.API_CONFIG else 20
        if args.database_name and ('New York' in args.database_name or 'Austin' in args.database_name):
            max_workers = min(max_workers, 50)
    
    print(f"=== Processing Database: {args.database_name} ===")
    print(f"SQL skeleton file: {skeleton_file}")
    print(f"Schema file: {schema_file}")
    print(f"Graph file directory: {graph_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Maximum retries: {args.max_retries}")
    print(f"Max workers: {max_workers}")
    if args.target_count:
        print(f"Target count: {args.target_count}")
    print()
    
    # If target_count specified, check current count first
    if args.target_count:
        single_output_path = os.path.join(output_dir, 'single', args.database_name)
        if os.path.exists(single_output_path):
            existing_count = len([f for f in os.listdir(single_output_path) 
                                 if f.startswith('generated_sql_') and f.endswith('.json')])
            needed = args.target_count - existing_count
            if needed <= 0:
                print(f"Already have {existing_count} SQLs, target is {args.target_count}. No need to generate more.")
                return
            print(f"Current: {existing_count}, Target: {args.target_count}, Need to generate: {needed}")
    
    # Process database
    # Determine max_workers
    max_workers = args.max_workers
    if max_workers is None:
        # Get from config, default 20 (lower concurrency to avoid resource issues)
        max_workers = fill_module.API_CONFIG.get('max_workers', 20) if fill_module.API_CONFIG else 20
        # Lower concurrency for large databases
        if args.database_name and ('New York' in args.database_name or 'Austin' in args.database_name):
            max_workers = min(max_workers, 50)  # Limit concurrency for large databases
    
    success_count, fail_count = fill_module.process_single_database(
        args.database_name,
        skeleton_file,
        schema_file,
        graph_dir,
        output_dir,
        max_retries=args.max_retries,
        max_workers=max_workers
    )
    
    print()
    print(f"=== Processing Complete ===")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total: {success_count + fail_count}")

if __name__ == '__main__':
    main()

