#!/usr/bin/env python3
"""
Clean up JOIN SQL files without results for 3- and 4-database queries
"""

import os
import json
import argparse
from collections import defaultdict

def cleanup_failed_sqls(sql_dir, db_counts=[3, 4], dry_run=False):
    """Clean up SQL files without results for the specified database count category"""
    
    if not os.path.exists(sql_dir):
        print(f"Directory does not exist: {sql_dir}")
        return
    
    files_to_delete = []
    stats = defaultdict(lambda: {'total': 0, 'with_results': 0, 'no_results': 0})
    
    print("=" * 70)
    print("Clean up JOIN SQL files without results for 3- and 4-database queries")
    print("=" * 70)
    print(f"SQL directory: {sql_dir}")
    print(f"Target database counts: {db_counts}")
    if dry_run:
        print("Warning: This is preview mode (dry-run), will not actually delete files")
    print()
    
    # Scan all SQL files
    for f in os.listdir(sql_dir):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    num_databases = data.get('metadata', {}).get('num_databases', 
                                                                len(data.get('databases', [])))
                    
                    # Only process files for specified database count
                    if num_databases in db_counts:
                        stats[num_databases]['total'] += 1
                        
                        if results is not None and len(results) > 0:
                            stats[num_databases]['with_results'] += 1
                        else:
                            stats[num_databases]['no_results'] += 1
                            files_to_delete.append((file_path, f, num_databases))
            except Exception as e:
                print(f"  Warning: failed to read file {f}: {e}")
    
    # Show statistics
    print("Statistics:")
    print("-" * 70)
    total_all = 0
    with_results_all = 0
    no_results_all = 0
    
    for db_count in sorted(stats.keys()):
        stat = stats[db_count]
        total_all += stat['total']
        with_results_all += stat['with_results']
        no_results_all += stat['no_results']
        
        print(f"  {db_count} databases:")
        print(f"    Total: {stat['total']}")
        print(f"    With results: {stat['with_results']}")
        print(f"    Without results: {stat['no_results']} (will be deleted)")
        print()
    
    print("-" * 70)
    print(f"Total:")
    print(f"  Total: {total_all}")
    print(f"  With results: {with_results_all}")
    print(f"  Without results: {no_results_all} (will be deleted)")
    print()
    
    if len(files_to_delete) == 0:
        print("No files need to be deleted")
        return
    
    # Show files to be deleted (first 10 and last 10)
    print(f"Will delete {len(files_to_delete)} files:")
    if len(files_to_delete) <= 20:
        for file_path, f, db_count in files_to_delete:
            print(f"  [{db_count}DB] {f}")
    else:
        for file_path, f, db_count in files_to_delete[:10]:
            print(f"  [{db_count}DB] {f}")
        print(f"  ... (omitted {len(files_to_delete) - 20}) ...")
        for file_path, f, db_count in files_to_delete[-10:]:
            print(f"  [{db_count}DB] {f}")
    print()
    
    # Execute deletion
    if dry_run:
        print("Preview mode: above files would be deleted (but not actually deleted)")
    else:
        deleted_count = 0
        failed_count = 0
        
        for file_path, f, db_count in files_to_delete:
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                print(f"  Delete failed: {f} - {e}")
                failed_count += 1
        
        print(f"\nDeleted {deleted_count} files")
        if failed_count > 0:
            print(f"Warning: delete failed for {failed_count} files")
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Clean up JOIN SQL files without results for 3- and 4-database queries')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL file directory')
    parser.add_argument('--only_3db', action='store_true',
                       help='Only clean up SQL for 3 databases')
    parser.add_argument('--only_4db', action='store_true',
                       help='Only clean up SQL for 4 databases')
    parser.add_argument('--dry_run', action='store_true',
                       help='Preview mode, do not actually delete files')
    
    args = parser.parse_args()
    
    # Determine database counts to clean up
    if args.only_3db:
        db_counts = [3]
    elif args.only_4db:
        db_counts = [4]
    else:
        db_counts = [3, 4]
    
    # Convert to absolute path
    if not os.path.isabs(args.sql_dir):
        # Get project root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        args.sql_dir = os.path.join(project_root, args.sql_dir)
    
    cleanup_failed_sqls(args.sql_dir, db_counts, args.dry_run)

if __name__ == '__main__':
    main()
