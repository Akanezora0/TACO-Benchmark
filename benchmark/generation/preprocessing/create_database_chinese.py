#!/usr/bin/env python3
"""
Create databases using Chinese table names (optimized version)

Key improvements:
1. Use original CSV filenames (without .csv suffix) as table names without pinyin conversion
2. Wrap all table and column names in double quotes so SQLite handles Chinese and special characters correctly
3. Generate standard-format schema JSON files for downstream use
"""

import os
import sqlite3
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

def quote_identifier(identifier):
    """Wrap identifier in double quotes so SQLite handles Chinese and special characters correctly"""
    # Escape double quotes
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'

def csv_to_sqlite_chinese(csv_folder_path, sqlite_db_path):
    """
    Convert CSV files to SQLite tables using Chinese table names
    
    Key improvements:
    - Use CSV filename (without .csv suffix) as table name without pinyin conversion
    - Wrap table and column names in double quotes so SQLite handles Chinese correctly
    """
    # Create SQLite database connection
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # Store database structure
    db_structure = {}
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    
    print(f"Found {len(csv_files)} CSV files")

    # Iterate CSV files and convert each to a SQLite table
    for file_name in tqdm(csv_files, desc="Processing CSV files"):
        csv_path = os.path.join(csv_folder_path, file_name)
        try:
            # Read CSV file
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Replace NaN with empty strings
            df = df.fillna('')

            # Get table name (remove .csv suffix, use original Chinese name)
            table_name = file_name.replace('.csv', '')
            
            # Wrap table name in double quotes so SQLite handles Chinese correctly
            quoted_table_name = quote_identifier(table_name)
            
            # Create table manually (wrap table and column names in double quotes)
            # Drop existing table first
            cursor.execute(f'DROP TABLE IF EXISTS {quoted_table_name}')
            
            # Build CREATE TABLE statement
            column_defs = []
            for col in df.columns:
                # Infer data type
                col_type = 'TEXT'  # default type
                if df[col].dtype == 'int64':
                    col_type = 'INTEGER'
                elif df[col].dtype == 'float64':
                    col_type = 'REAL'
                
                quoted_col = quote_identifier(col)
                column_defs.append(f'{quoted_col} {col_type}')
            
            create_table_sql = f'CREATE TABLE {quoted_table_name} ({", ".join(column_defs)})'
            cursor.execute(create_table_sql)
            
            # Insert data
            for _, row in df.iterrows():
                # Build INSERT statement with quoted table and column names
                columns = ', '.join([quote_identifier(col) for col in df.columns])
                placeholders = ', '.join(['?' for _ in df.columns])
                insert_sql = f'INSERT INTO {quoted_table_name} ({columns}) VALUES ({placeholders})'
                cursor.execute(insert_sql, tuple(row))
            
            # Get column names and store in db_structure
            db_structure[table_name] = {
                'columns': df.columns.tolist(),
                'row_count': len(df)
            }
            
        except Exception as e:
            # On error, skip file and print error message
            print(f"\nError processing file {csv_path}: {e}")
            continue  # skip current file and continue with next

    # Commit and close database connection
    conn.commit()
    conn.close()

    # Return database structure
    return db_structure

def save_schema_as_json(db_structure, db_folder_path, db_name):
    """
    Save standard-format schema JSON file
    
    Format:
    {
        "tables": [
            {
                "table_name": "<table_name>",
                "table_comment": "<table_name>",
                "table_description": "No description available.",
                "columns": [
                    {
                        "column_name": "<column_name>",
                        "data_type": "TEXT"
                    }
                ],
                "primary_keys": [],
                "foreign_keys": []
            }
        ]
    }
    """
    schema = {'tables': []}
    
    for table_name, table_info in db_structure.items():
        columns = []
        for col_name in table_info['columns']:
            # Default type is TEXT; can be inferred if needed
            columns.append({
                'column_name': col_name,
                'data_type': 'TEXT'
            })
        
        schema['tables'].append({
            'table_name': table_name,
            'table_comment': table_name,
            'table_description': 'No description available.',
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        })
    
    # Save schema JSON file
    schema_file_path = os.path.join(db_folder_path, f"{db_name}.json")
    try:
        with open(schema_file_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        print(f"Schema saved: {schema_file_path}")
    except Exception as e:
        print(f"Error saving schema file {schema_file_path}: {e}")

def process_parsed_data_to_sqlite_chinese(parsed_data_dir, database_dir):
    """
    Process all folders in parsed_data and generate SQLite databases with Chinese table names
    """
    # Iterate all folders under parsed_data/
    folders = [f for f in os.listdir(parsed_data_dir) 
               if os.path.isdir(os.path.join(parsed_data_dir, f))]
    
    print(f"Found {len(folders)} database folders")
    
    for folder_name in tqdm(folders, desc="Processing databases"):
        folder_path = os.path.join(parsed_data_dir, folder_name)
        try:
            # Create a SQLite database for each folder
            db_folder_path = os.path.join(database_dir, folder_name)
            os.makedirs(db_folder_path, exist_ok=True)
            
            sqlite_db_path = os.path.join(db_folder_path, f"{folder_name}.db")
            
            # Convert all CSV files in folder to SQLite tables using Chinese table names
            db_structure = csv_to_sqlite_chinese(folder_path, sqlite_db_path)
            
            # Save standard-format schema JSON file
            save_schema_as_json(db_structure, db_folder_path, folder_name)
            
            print(f"\n✓ Database '{folder_name}' created")
            print(f"  Table count: {len(db_structure)}")
            print(f"  Database file: {sqlite_db_path}")
            
        except Exception as e:
            # On folder processing error, print message and continue
            print(f"\n✗ Error processing folder {folder_path}: {e}")
            continue  # skip current folder and continue with next

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Create SQLite databases using Chinese table names')
    project_root = Path(__file__).resolve().parents[3]
    
    parser.add_argument('--parsed_data_dir', type=str, default=None,
                       help='parsed_data directory (default: ../../data/parsed_data)')
    parser.add_argument('--database_dir', type=str, default=None,
                       help='Database output directory (default: ../../data/database_chinese)')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.parsed_data_dir is None:
        args.parsed_data_dir = str(project_root / 'benchmark' / 'data' / 'beijing' / 'parsed_data')
    if args.database_dir is None:
        args.database_dir = str(project_root / 'benchmark' / 'data' / 'beijing' / 'database_chinese')
    
    # Convert to absolute paths
    args.parsed_data_dir = os.path.abspath(args.parsed_data_dir)
    args.database_dir = os.path.abspath(args.database_dir)
    
    # Create output directory
    os.makedirs(args.database_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("Create SQLite databases using Chinese table names")
    print(f"{'='*60}")
    print(f"Input directory: {args.parsed_data_dir}")
    print(f"Output directory: {args.database_dir}")
    print(f"{'='*60}\n")
    
    # Process all folders in parsed_data and generate SQLite databases
    process_parsed_data_to_sqlite_chinese(args.parsed_data_dir, args.database_dir)
    
    print(f"\n{'='*60}")
    print("✓ All databases created!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
