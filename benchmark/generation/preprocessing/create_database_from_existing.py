#!/usr/bin/env python3
"""
Create databases with Chinese table names from existing database JSON files

Because there is no parsed_data directory, we can:
1. Read data from existing database JSON files
2. Use table_name_mappings.json to reverse-lookup original Chinese table names
3. Create new databases that use Chinese table names
"""

import os
import sqlite3
import json
from tqdm import tqdm

def quote_identifier(identifier):
    """Wrap identifier in double quotes so SQLite handles Chinese and special characters correctly"""
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'

def load_table_name_mappings(mappings_file):
    """Load table name mappings for reverse lookup of original Chinese table names"""
    if not os.path.exists(mappings_file):
        return {}
    
    with open(mappings_file, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    # Build reverse mapping: {database_name: {pinyin_table_name: original_csv_filename}}
    reverse_mappings = {}
    for db_name, db_mappings in mappings.items():
        reverse_mappings[db_name] = {v: k for k, v in db_mappings.items()}
    
    return reverse_mappings

def create_database_from_json(json_file, db_file, reverse_mappings, db_name):
    """Create a database with Chinese table names from a JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Get reverse mapping for this database
    db_reverse_mapping = reverse_mappings.get(db_name, {})
    
    created_tables = 0
    
    for pinyin_table_name, table_data in tqdm(data.items(), desc=f"Processing {db_name}", leave=False):
        # Try to get original Chinese table name from reverse mapping
        original_csv_name = db_reverse_mapping.get(pinyin_table_name, None)
        
        if original_csv_name:
            # Remove .csv suffix to get original table name
            original_table_name = original_csv_name.replace('.csv', '')
        else:
            # If no mapping exists, skip this table (possibly dirty data or incomplete mapping)
            print(f"\nWarning: table '{pinyin_table_name}' not found in mapping, skipping")
            continue
        
        # Wrap table name in double quotes
        quoted_table_name = quote_identifier(original_table_name)
        
        # Get column info
        columns = table_data.get('columns', [])
        if not columns:
            continue
        
        # Build CREATE TABLE statement
        column_defs = []
        for col in columns:
            quoted_col = quote_identifier(col)
            column_defs.append(f'{quoted_col} TEXT')  # default to TEXT type
        
        create_table_sql = f'CREATE TABLE {quoted_table_name} ({", ".join(column_defs)})'
        
        try:
            cursor.execute(f'DROP TABLE IF EXISTS {quoted_table_name}')
            cursor.execute(create_table_sql)
            
            # Insert data
            rows = table_data.get('data', [])
            if rows:
                columns_str = ', '.join([quote_identifier(col) for col in columns])
                placeholders = ', '.join(['?' for _ in columns])
                insert_sql = f'INSERT INTO {quoted_table_name} ({columns_str}) VALUES ({placeholders})'
                
                for row in rows:
                    values = [row.get(col, '') for col in columns]
                    cursor.execute(insert_sql, tuple(values))
            
            created_tables += 1
        except Exception as e:
            print(f"\nError creating table {original_table_name}: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    return created_tables

def save_schema_as_json(data, db_folder_path, db_name, reverse_mappings):
    """Save standard-format schema JSON file"""
    db_reverse_mapping = reverse_mappings.get(db_name, {})
    schema = {'tables': []}
    
    for pinyin_table_name, table_data in data.items():
        # Get original Chinese table name
        original_csv_name = db_reverse_mapping.get(pinyin_table_name, None)
        if original_csv_name:
            original_table_name = original_csv_name.replace('.csv', '')
        else:
            original_table_name = pinyin_table_name
        
        columns = []
        for col_name in table_data.get('columns', []):
            columns.append({
                'column_name': col_name,
                'data_type': 'TEXT'
            })
        
        schema['tables'].append({
            'table_name': original_table_name,
            'table_comment': original_table_name,
            'table_description': 'No description available.',
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        })
    
    schema_file_path = os.path.join(db_folder_path, f"{db_name}.json")
    with open(schema_file_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    
    return schema_file_path

def process_existing_databases(database_dir, output_dir, mappings_file):
    """Create new databases with Chinese table names from existing databases"""
    # Load table name mappings
    reverse_mappings = load_table_name_mappings(mappings_file)
    
    # Get all database folders
    db_folders = [f for f in os.listdir(database_dir) 
                  if os.path.isdir(os.path.join(database_dir, f))]
    
    print(f"Found {len(db_folders)} databases")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for db_name in tqdm(db_folders, desc="Processing databases"):
        db_folder = os.path.join(database_dir, db_name)
        json_file = os.path.join(db_folder, f"{db_name}.json")
        
        if not os.path.exists(json_file):
            print(f"\nWarning: skipping {db_name}, JSON file not found")
            continue
        
        try:
            # Read existing database JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create output directory
            output_db_folder = os.path.join(output_dir, db_name)
            os.makedirs(output_db_folder, exist_ok=True)
            
            # Create database file
            db_file = os.path.join(output_db_folder, f"{db_name}.db")
            created_tables = create_database_from_json(json_file, db_file, reverse_mappings, db_name)
            
            # Save schema JSON
            schema_file = save_schema_as_json(data, output_db_folder, db_name, reverse_mappings)
            
            print(f"\n✓ {db_name}: created {created_tables} tables")
            print(f"  Database: {db_file}")
            print(f"  Schema: {schema_file}")
            
        except Exception as e:
            print(f"\n✗ Error processing {db_name}: {e}")
            continue

def main():
    import argparse
    
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Create databases with Chinese table names from existing databases')
    project_root = Path(__file__).resolve().parents[3]
    
    parser.add_argument('--database_dir', type=str, default=None,
                       help='Existing database directory (default: ../../data/beijing/database)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../../data/beijing/database_chinese)')
    parser.add_argument('--mappings_file', type=str, default=None,
                       help='Table name mapping file (default: benchmark/data/table_name_mappings.json)')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.database_dir is None:
        args.database_dir = str(project_root / 'benchmark' / 'data' / 'beijing' / 'database')
    if args.output_dir is None:
        args.output_dir = str(project_root / 'benchmark' / 'data' / 'beijing' / 'database_chinese')
    if args.mappings_file is None:
        args.mappings_file = str(project_root / 'benchmark' / 'data' / 'table_name_mappings.json')
    
    # Convert to absolute paths
    args.database_dir = os.path.abspath(args.database_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.mappings_file = os.path.abspath(args.mappings_file)
    
    print(f"{'='*60}")
    print("Create databases with Chinese table names from existing databases")
    print(f"{'='*60}")
    print(f"Input directory: {args.database_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mapping file: {args.mappings_file}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(args.mappings_file):
        print(f"Warning: mapping file not found {args.mappings_file}")
        print("Will use pinyin table names (not recommended)")
    
    process_existing_databases(args.database_dir, args.output_dir, args.mappings_file)
    
    print(f"\n{'='*60}")
    print("✓ All databases created!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
