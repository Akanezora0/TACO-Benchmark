#!/usr/bin/env python3
"""
Extract database schema (Chinese table name version)

Extract standard-format schema directly from database JSON files or SQLite databases,
using original Chinese table names.
"""

import os
import json
import sqlite3
import argparse
from tqdm import tqdm

def extract_schema_from_json(json_file):
    """Extract schema from a database JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    schema = {'tables': []}
    
    # Database JSON format: {table_name: {columns: [...], data: [...]}}
    for table_name, table_data in data.items():
        columns = []
        if 'columns' in table_data:
            for col_name in table_data['columns']:
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'  # default type; can be inferred if needed
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

def extract_schema_from_db(db_file):
    """Extract schema from a SQLite database file"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    schema = {'tables': []}
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for (table_name,) in tables:
        # Get column info for the table
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns_info = cursor.fetchall()
        
        columns = []
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]  # SQLite type
            columns.append({
                'column_name': col_name,
                'data_type': col_type
            })
        
        schema['tables'].append({
            'table_name': table_name,
            'table_comment': table_name,
            'table_description': 'No description available.',
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        })
    
    conn.close()
    return schema

def process_databases(database_dir, output_dir):
    """Process all databases and extract schema"""
    # Get all database folders
    db_folders = [f for f in os.listdir(database_dir) 
                  if os.path.isdir(os.path.join(database_dir, f))]
    
    print(f"Found {len(db_folders)} databases")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for db_name in tqdm(db_folders, desc="Extracting schema"):
        db_folder = os.path.join(database_dir, db_name)
        
        # Try extracting from JSON file
        json_file = os.path.join(db_folder, f"{db_name}.json")
        db_file = os.path.join(db_folder, f"{db_name}.db")
        
        schema = None
        if os.path.exists(json_file):
            try:
                schema = extract_schema_from_json(json_file)
            except Exception as e:
                print(f"\nFailed to extract schema from JSON {json_file}: {e}")
        
        if schema is None and os.path.exists(db_file):
            try:
                schema = extract_schema_from_db(db_file)
            except Exception as e:
                print(f"\nFailed to extract schema from database {db_file}: {e}")
        
        if schema:
            # Save schema file
            schema_file = os.path.join(output_dir, f"{db_name}_schema.json")
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
        else:
            print(f"\nWarning: unable to extract schema for database '{db_name}'")

def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Extract database schema (Chinese table names)')
    project_root = Path(__file__).resolve().parents[3]
    
    parser.add_argument('--database_dir', type=str, default=None,
                       help='Database directory (default: ../../data/beijing/database_chinese)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Schema output directory (default: ../../data/beijing/schema_chinese)')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.database_dir is None:
        args.database_dir = str(project_root / 'benchmark' / 'data' / 'beijing' / 'database_chinese')
    if args.output_dir is None:
        args.output_dir = str(project_root / 'benchmark' / 'data' / 'beijing' / 'schema_chinese')
    
    # Convert to absolute paths
    args.database_dir = os.path.abspath(args.database_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    print(f"{'='*60}")
    print("Extract database schema (Chinese table names)")
    print(f"{'='*60}")
    print(f"Database directory: {args.database_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    process_databases(args.database_dir, args.output_dir)
    
    print(f"\n{'='*60}")
    print("✓ All schema extraction complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
