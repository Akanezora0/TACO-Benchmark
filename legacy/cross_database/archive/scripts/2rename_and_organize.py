#!/usr/bin/env python3
"""
Rename and organize SQL files that have results.

Classify by database count (2, 3, or 4 databases), then rename sequentially.
For example: cross_db_generated_sql_0.json, cross_db_generated_sql_1.json, ...
"""

import os
import json
import argparse
from collections import defaultdict

def rename_and_organize(sql_dir):
    """Rename and organize SQL files."""
    
    # 1. Collect all SQL files with results, grouped by database count
    sqls_by_db_count = defaultdict(list)
    
    print("Collecting SQL files with results...")
    for f in sorted(os.listdir(sql_dir)):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    
                    # Only process files with results
                    if results is not None and len(results) > 0:
                        num_databases = data.get('metadata', {}).get('num_databases', 
                                                                    len(data.get('databases', [])))
                        sqls_by_db_count[num_databases].append((file_path, data))
            except Exception as e:
                print(f"Error processing file {f}: {e}")
    
    print(f"\nGrouped by database count:")
    for db_count in sorted(sqls_by_db_count.keys()):
        print(f"  {db_count} databases: {len(sqls_by_db_count[db_count])} files")
    
    # 2. Rename within each database-count category
    total_renamed = 0
    rename_map = {}
    
    for db_count in sorted(sqls_by_db_count.keys()):
        files = sqls_by_db_count[db_count]
        
        # Sort by some order (modification time or SQL content)
        # Here we simply sort by file path
        files.sort(key=lambda x: x[0])
        
        # Compute starting index (based on existing files)
        existing_indices = []
        for f in os.listdir(sql_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                    existing_indices.append(idx)
                except:
                    pass
        
        start_idx = max(existing_indices) + 1 if existing_indices else 0
        
        # Rename
        for i, (file_path, data) in enumerate(files):
            new_name = f"cross_db_generated_sql_{start_idx + i}.json"
            new_path = os.path.join(sql_dir, new_name)
            
            # Skip if target name already exists (avoid overwrite)
            if os.path.exists(new_path) and new_path != file_path:
                continue
            
            # Skip if file already has the correct name
            if os.path.basename(file_path) == new_name:
                continue
            
            rename_map[file_path] = new_path
            total_renamed += 1
    
    # 3. Perform renames (temp name first, then final name, to avoid conflicts)
    print(f"\nRenaming {total_renamed} files...")
    
    # Rename to temporary names first
    temp_map = {}
    for old_path, new_path in rename_map.items():
        temp_name = os.path.join(os.path.dirname(old_path), f"__temp_{os.path.basename(new_path)}")
        os.rename(old_path, temp_name)
        temp_map[temp_name] = new_path
    
    # Then rename to final names
    for temp_path, final_path in temp_map.items():
        os.rename(temp_path, final_path)
    
    print(f"Rename complete: {total_renamed} files")
    
    return total_renamed

def main():
    parser = argparse.ArgumentParser(description='Rename and organize SQL files with results')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL file directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Renaming and organizing SQL files")
    print("=" * 70)
    print()
    
    rename_and_organize(args.sql_dir)
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()

