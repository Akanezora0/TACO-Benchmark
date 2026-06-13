#!/usr/bin/env python3
"""
Statistics on JOIN SQL involving 2, 3, and 4 databases
"""

import os
import json
import argparse
from collections import defaultdict

def count_databases_in_sql(data):
    """Extract number of databases involved from SQL data"""
    # Method 1: get from databases field
    if 'databases' in data:
        databases = data['databases']
        if isinstance(databases, list):
            return len(set(databases))
        elif isinstance(databases, dict):
            return len(databases)
    
    # Method 2: get from table_database_mapping
    if 'table_database_mapping' in data:
        mapping = data['table_database_mapping']
        if isinstance(mapping, dict):
            databases = set()
            for table, db in mapping.items():
                if isinstance(db, str):
                    databases.add(db)
                elif isinstance(db, dict) and 'database' in db:
                    databases.add(db['database'])
            return len(databases)
    
    # Method 3: get from schema_graphs
    if 'schema_graphs' in data:
        schema_graphs = data['schema_graphs']
        if isinstance(schema_graphs, dict):
            return len(schema_graphs)
    
    return 0

def statistics_join_sqls(backup_dir):
    """Statistics on database count distribution for JOIN SQL"""
    
    if not os.path.exists(backup_dir):
        print(f"Backup directory does not exist: {backup_dir}")
        return
    
    files = [f for f in os.listdir(backup_dir) 
             if f.startswith('cross_db_generated_sql_') and f.endswith('.json')]
    
    print(f"Found {len(files)} SQL files")
    print()
    
    # Statistics
    db_count_stats = defaultdict(int)  # database count -> file count
    db_count_details = defaultdict(list)  # database count -> file list
    
    valid_files = 0
    invalid_files = 0
    
    for f in files:
        file_path = os.path.join(backup_dir, f)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Check if has results
                results = data.get('results', [])
                if results is None or len(results) == 0:
                    continue
                
                db_count = count_databases_in_sql(data)
                
                if db_count > 0:
                    db_count_stats[db_count] += 1
                    db_count_details[db_count].append(f)
                    valid_files += 1
                else:
                    invalid_files += 1
                    
        except Exception as e:
            print(f"Failed to read file {f}: {e}")
            invalid_files += 1
    
    # Output statistics
    print("=" * 70)
    print("JOIN SQL database count statistics")
    print("=" * 70)
    print(f"Valid file count: {valid_files}")
    if invalid_files > 0:
        print(f"Invalid file count: {invalid_files}")
    print()
    
    print("Database count distribution:")
    print("-" * 70)
    total = 0
    for db_count in sorted(db_count_stats.keys()):
        count = db_count_stats[db_count]
        total += count
        percentage = (count / valid_files * 100) if valid_files > 0 else 0
        print(f"  {db_count} databases: {count} SQL ({percentage:.1f}%)")
    
    print("-" * 70)
    print(f"  Total: {total} SQL")
    print()
    
    # Detail files for 2, 3, and 4 databases
    for db_count in [2, 3, 4]:
        if db_count in db_count_details:
            files_list = db_count_details[db_count]
            print(f"{db_count}-database SQL files ({len(files_list)}):")
            # Show first 10 and last 10 only
            if len(files_list) <= 20:
                for f in files_list:
                    print(f"  - {f}")
            else:
                for f in files_list[:10]:
                    print(f"  - {f}")
                print(f"  ... (omitted {len(files_list) - 20}) ...")
                for f in files_list[-10:]:
                    print(f"  - {f}")
            print()
    
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Statistics on JOIN SQL involving 2, 3, and 4 databases')
    parser.add_argument('--backup_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join_backup_51',
                       help='Backup directory')
    
    args = parser.parse_args()
    
    statistics_join_sqls(args.backup_dir)

if __name__ == '__main__':
    main()
