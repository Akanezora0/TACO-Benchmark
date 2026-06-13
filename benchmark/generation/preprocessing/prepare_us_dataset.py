#!/usr/bin/env python3
"""
Prepare the US dataset following the Beijing dataset workflow

Steps:
1. Create the US dataset directory structure
2. Convert database format (from old format to standard format)
3. Extract SQL skeletons and group by database
4. Prepare for graph generation and SQL filling
"""

import os
import json
import shutil
import sqlite3
from pathlib import Path

def convert_us_database_format(old_db_path, old_json_path, new_db_dir, db_name):
    """
    Convert US dataset database format
    
    From old format (top-level keys are table names) to standard format (with tables key)
    """
    # Read old-format JSON
    with open(old_json_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    # Convert to standard format
    schema = {'tables': []}
    
    for table_name, table_data in old_data.items():
        if isinstance(table_data, dict) and 'columns' in table_data:
            columns = []
            for col_name in table_data['columns']:
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'  # default type
                })
            
            schema['tables'].append({
                'table_name': table_name,
                'table_comment': table_name,
                'table_description': 'No description available.',
                'columns': columns,
                'primary_keys': [],
                'foreign_keys': []
            })
    
    # Create new directory
    os.makedirs(new_db_dir, exist_ok=True)
    
    # Save standard-format schema JSON
    schema_file = os.path.join(new_db_dir, f"{db_name}.json")
    with open(schema_file, 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    
    # Copy database file
    new_db_file = os.path.join(new_db_dir, f"{db_name}.db")
    if os.path.exists(old_db_path):
        shutil.copy2(old_db_path, new_db_file)
    
    return schema_file, new_db_file

def extract_sql_skeletons_by_database(skeleton_file, output_dir):
    """
    Extract SQL skeletons from new_sql_skeletons.json and group by database
    
    Database names must be inferred from SQL (via table names)
    """
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    
    # Extract table names from SQL and infer database
    # This depends on the actual SQL format
    # For now, group by table name first
    
    db_skeletons = {}
    
    for item in skeletons:
        if isinstance(item, dict):
            sql = item.get('sql', '')
            sql_framework = item.get('sql_framework', '')
            
            # Extract table name from SQL (first identifier after FROM)
            if 'FROM' in sql.upper():
                parts = sql.upper().split('FROM')
                if len(parts) > 1:
                    table_part = parts[1].strip().split()[0].strip(';')
                    table_name = table_part.strip()
                    
                    # Infer database from table name (requires table-to-database mapping)
                    # For now, use table name as database name (adjust later based on actual data)
                    db_name = table_name  # temporary approach
                    
                    if db_name not in db_skeletons:
                        db_skeletons[db_name] = []
                    
                    db_skeletons[db_name].append({
                        'sql_framework': sql_framework,
                        'sql': sql
                    })
    
    # Save SQL skeletons grouped by database
    os.makedirs(output_dir, exist_ok=True)
    
    for db_name, skeletons in db_skeletons.items():
        skeleton_file = os.path.join(output_dir, f"{db_name}_sql_skeleton.json")
        with open(skeleton_file, 'w', encoding='utf-8') as f:
            json.dump(skeletons, f, ensure_ascii=False, indent=2)
    
    return db_skeletons

def main():
    # Set paths
    project_root = Path(__file__).resolve().parents[3]
    old_america_dir = project_root / 'old' / 'saturn' / 'America' / 'data'
    old_us_dir = project_root / 'old' / 'saturn' / 'TACO-Benchmark' / 'us' / 'data'
    new_us_dir = project_root / 'benchmark' / 'data' / 'us'
    
    # Create new directory structure
    new_db_dir = new_us_dir / 'database_chinese'
    new_skeleton_dir = new_us_dir / 'output' / 'sql_skeleton'
    new_graph_dir = new_us_dir / 'output' / 'graph_chinese'
    new_output_dir = new_us_dir / 'output' / 'single'
    
    for dir_path in [new_db_dir, new_skeleton_dir, new_graph_dir, new_output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("=== Preparing US dataset ===\n")
    
    # Step 1: convert database format
    print("Step 1: Converting database format...")
    old_db_dir = old_america_dir / 'database'
    
    if old_db_dir.exists():
        db_count = 0
        for db_name in os.listdir(old_db_dir):
            db_path = old_db_dir / db_name
            if db_path.is_dir():
                # Find .db and .json files
                db_files = list(db_path.glob('*.db'))
                json_files = list(db_path.glob('*.json'))
                
                if db_files and json_files:
                    old_db_file = db_files[0]
                    old_json_file = json_files[0]
                    
                    # Create new database directory (use safe directory name)
                    safe_db_name = db_name.replace('/', '_').replace('\\', '_')
                    new_db_subdir = new_db_dir / safe_db_name
                    
                    try:
                        schema_file, new_db_file = convert_us_database_format(
                            str(old_db_file), str(old_json_file),
                            str(new_db_subdir), safe_db_name
                        )
                        db_count += 1
                        print(f"  ✓ {safe_db_name[:60]}...")
                    except Exception as e:
                        print(f"  ✗ {safe_db_name[:60]}... Error: {e}")
        
        print(f"\nConversion complete: {db_count} databases\n")
    
    # Step 2: extract SQL skeletons
    print("Step 2: Extracting SQL skeletons...")
    old_skeleton_file = old_us_dir.parent / 'new_sql_skeletons.json'
    
    if old_skeleton_file.exists():
        try:
            db_skeletons = extract_sql_skeletons_by_database(
                str(old_skeleton_file), str(new_skeleton_dir)
            )
            print(f"  Extracted SQL skeletons for {len(db_skeletons)} databases")
        except Exception as e:
            print(f"  Extraction failed: {e}")
    
    print("\n=== US dataset preparation complete ===")
    print(f"Database directory: {new_db_dir}")
    print(f"SQL skeleton directory: {new_skeleton_dir}")
    print(f"Graph directory: {new_graph_dir}")
    print(f"Output directory: {new_output_dir}")

if __name__ == '__main__':
    main()
