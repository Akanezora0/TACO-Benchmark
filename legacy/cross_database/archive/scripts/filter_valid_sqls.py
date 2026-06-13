#!/usr/bin/env python3
"""
Filter out non-executable cross-database SQL and keep only entries with execution results.
"""

import json
import os
import shutil
from collections import defaultdict

def filter_valid_sqls(sql_dir, output_dir=None):
    """
    Filter valid SQL files (those with execution results).
    
    Args:
        sql_dir: SQL result directory
        output_dir: Output directory (if None, overwrite in place)
    """
    if not os.path.exists(sql_dir):
        print(f"Directory does not exist: {sql_dir}")
        return
    
    sql_files = sorted([f for f in os.listdir(sql_dir) 
                       if f.startswith('cross_db_generated_sql_') and f.endswith('.json')])
    
    print(f"=" * 70)
    print(f"Filtering cross-database SQL results")
    print(f"=" * 70)
    print(f"\nTotal files: {len(sql_files)}")
    
    valid_files = []
    invalid_files = []
    
    for sql_file in sql_files:
        file_path = os.path.join(sql_dir, sql_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            error = data.get('metadata', {}).get('execution_error')
            
            # Valid if has results and no error
            if len(results) > 0 and not error:
                valid_files.append(sql_file)
            else:
                invalid_files.append(sql_file)
        except Exception as e:
            print(f"Failed to read file {sql_file}: {e}")
            invalid_files.append(sql_file)
    
    print(f"\nValid files (with execution results): {len(valid_files)} ({len(valid_files)/len(sql_files)*100:.1f}%)")
    print(f"Invalid files (no execution results): {len(invalid_files)} ({len(invalid_files)/len(sql_files)*100:.1f}%)")
    
    if output_dir:
        # Copy valid files to new directory
        os.makedirs(output_dir, exist_ok=True)
        for sql_file in valid_files:
            src = os.path.join(sql_dir, sql_file)
            dst = os.path.join(output_dir, sql_file)
            shutil.copy2(src, dst)
        print(f"\nValid files copied to: {output_dir}")
    else:
        # Delete invalid files
        print(f"\nDeleting invalid files...")
        for sql_file in invalid_files:
            file_path = os.path.join(sql_dir, sql_file)
            os.remove(file_path)
        print(f"Deleted {len(invalid_files)} invalid files")
        print(f"Kept {len(valid_files)} valid files")
    
    # Renumber (optional)
    if output_dir or len(invalid_files) > 0:
        print(f"\nRenumbering files...")
        valid_files_sorted = sorted(valid_files, key=lambda x: int(re.search(r'(\d+)', x).group(1)) if re.search(r'(\d+)', x) else 0)
        
        for i, sql_file in enumerate(valid_files_sorted):
            old_path = os.path.join(sql_dir, sql_file)
            new_name = f"cross_db_generated_sql_{i}.json"
            new_path = os.path.join(sql_dir, new_name)
            
            if old_path != new_path:
                os.rename(old_path, new_path)
        
        print(f"Renumbered {len(valid_files_sorted)} files")
    
    return len(valid_files), len(invalid_files)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Filter valid cross-database SQL')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single',
                       help='SQL result directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (if set, copy valid files; otherwise delete invalid files)')
    
    args = parser.parse_args()
    
    valid_count, invalid_count = filter_valid_sqls(args.sql_dir, args.output_dir)
    
    print(f"\n" + "=" * 70)
    print(f"Done!")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {invalid_count}")
    print(f"=" * 70)

if __name__ == '__main__':
    import re
    main()

