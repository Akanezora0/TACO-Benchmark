#!/usr/bin/env python3
"""
Clean up SQL files without results

Delete all SQL files with no results or empty results
"""

import os
import json
import argparse

def cleanup_failed_sqls(sql_dir):
    """Clean up SQL files without results"""
    deleted_count = 0
    kept_count = 0
    
    print(f"Cleaning directory: {sql_dir}")
    
    for f in sorted(os.listdir(sql_dir)):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    
                    # Delete file if no results or results are empty
                    if results is None or len(results) == 0:
                        os.remove(file_path)
                        deleted_count += 1
                    else:
                        kept_count += 1
            except Exception as e:
                print(f"Error processing file {f}: {e}")
                # Also delete files that fail to read
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except:
                    pass
    
    print(f"\nCleanup complete:")
    print(f"  Kept: {kept_count} files")
    print(f"  Deleted: {deleted_count} files")
    
    return kept_count, deleted_count

def main():
    parser = argparse.ArgumentParser(description='Clean up SQL files without results')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL file directory')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Clean up SQL files without results")
    print("=" * 70)
    print()
    
    cleanup_failed_sqls(args.sql_dir)
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()

