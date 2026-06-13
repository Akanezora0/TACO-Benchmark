#!/usr/bin/env python3
"""
Back up existing results and clear the directory to prepare for regeneration.
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def backup_existing_results(sql_dir, backup_dir, start_index=0):
    """Back up existing SQL files with results and rename them with consecutive indices."""
    
    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)
    
    # Collect all SQL files that have results
    valid_files = []
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    # Files with results
                    if results is not None and len(results) > 0:
                        valid_files.append((file_path, f))
            except:
                pass
    
    print(f"Found {len(valid_files)} SQL files with results")
    
    # Sort by filename to ensure consistent order
    valid_files.sort(key=lambda x: x[1])
    
    # Back up and rename
    for idx, (file_path, original_name) in enumerate(valid_files):
        new_name = f"cross_db_generated_sql_{start_index + idx}.json"
        backup_path = os.path.join(backup_dir, new_name)
        shutil.copy2(file_path, backup_path)
        print(f"  Backed up: {original_name} -> {new_name}")
    
    return len(valid_files)

def clear_sql_directory(sql_dir):
    """Clear the SQL directory."""
    print(f"\nClearing directory: {sql_dir}")
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            os.remove(file_path)
            print(f"  Deleted: {f}")
    print("Directory cleared")

def main():
    parser = argparse.ArgumentParser(description='Back up existing results and clear the directory')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL file directory')
    parser.add_argument('--backup_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join_backup_51',
                       help='Backup directory')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Starting index')
    parser.add_argument('--clear', action='store_true',
                       help='Whether to clear the original directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Backing up existing results")
    print("=" * 70)
    
    # Back up
    count = backup_existing_results(args.sql_dir, args.backup_dir, args.start_index)
    print(f"\n✅ Backed up {count} files to {args.backup_dir}")
    
    # Clear
    if args.clear:
        clear_sql_directory(args.sql_dir)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()
