#!/usr/bin/env python3
"""
Analyze cross-database SQL execution results and filter valid SQL.
"""

import json
import os
import re
from collections import defaultdict

def analyze_execution_results(sql_dir):
    """Analyze execution results."""
    sql_files = sorted([f for f in os.listdir(sql_dir) 
                       if f.startswith('cross_db_generated_sql_') and f.endswith('.json')])
    
    print("=" * 70)
    print("Cross-database SQL execution result analysis")
    print("=" * 70)
    print(f"\nTotal files: {len(sql_files)}")
    
    has_results = 0
    no_results = 0
    error_types = defaultdict(int)
    
    for sql_file in sql_files:
        file_path = os.path.join(sql_dir, sql_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            error = data.get('metadata', {}).get('execution_error')
            
            if len(results) > 0:
                has_results += 1
            else:
                no_results += 1
                if error:
                    # Extract error type
                    if 'ATTACH' in error:
                        error_types['ATTACH failure'] += 1
                    elif 'single-database format' in error.lower():
                        error_types['Single-database format failure'] += 1
                    else:
                        error_types['Other error'] += 1
                else:
                    error_types['No error info'] += 1
        except Exception as e:
            no_results += 1
            error_types['File read failure'] += 1
    
    print(f"\nWith execution results: {has_results} ({has_results/len(sql_files)*100:.1f}%)")
    print(f"Without execution results: {no_results} ({no_results/len(sql_files)*100:.1f}%)")
    
    print(f"\nError type distribution:")
    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {error_type}: {count}")
    
    return has_results, no_results

def filter_valid_sqls(sql_dir):
    """Filter valid SQL and delete invalid entries."""
    sql_files = sorted([f for f in os.listdir(sql_dir) 
                       if f.startswith('cross_db_generated_sql_') and f.endswith('.json')])
    
    valid_files = []
    invalid_files = []
    
    for sql_file in sql_files:
        file_path = os.path.join(sql_dir, sql_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            error = data.get('metadata', {}).get('execution_error')
            
            # Valid: has results and no error
            if len(results) > 0 and not error:
                valid_files.append(sql_file)
            else:
                invalid_files.append(sql_file)
        except:
            invalid_files.append(sql_file)
    
    print(f"\n" + "=" * 70)
    print(f"Filter results")
    print(f"=" * 70)
    print(f"Valid files: {len(valid_files)}")
    print(f"Invalid files: {len(invalid_files)}")
    
    # Delete invalid files
    print(f"\nDeleting invalid files...")
    for sql_file in invalid_files:
        file_path = os.path.join(sql_dir, sql_file)
        os.remove(file_path)
    
    # Renumber
    print(f"Renumbering valid files...")
    valid_files_sorted = sorted(valid_files, 
                                key=lambda x: int(re.search(r'(\d+)', x).group(1)) 
                                if re.search(r'(\d+)', x) else 0)
    
    for i, sql_file in enumerate(valid_files_sorted):
        old_path = os.path.join(sql_dir, sql_file)
        new_name = f"cross_db_generated_sql_{i}.json"
        new_path = os.path.join(sql_dir, new_name)
        
        if old_path != new_path:
            os.rename(old_path, new_path)
    
    print(f"Done! Kept {len(valid_files_sorted)} valid files")
    
    return len(valid_files_sorted)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze and filter cross-database SQL')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single',
                       help='SQL result directory')
    parser.add_argument('--filter', action='store_true',
                       help='Whether to filter invalid SQL')
    
    args = parser.parse_args()
    
    # Analyze
    has_results, no_results = analyze_execution_results(args.sql_dir)
    
    # Filter if requested
    if args.filter:
        valid_count = filter_valid_sqls(args.sql_dir)
        print(f"\nFinal valid SQL count: {valid_count}")
        print(f"Target count: 359 (cross 2 databases)")
        if valid_count < 359:
            print(f"⚠️  Not enough valid SQL; need to generate more SQL skeletons")
            print(f"   At {has_results/(has_results+no_results)*100:.1f}% success rate, need roughly {int(359/(has_results/(has_results+no_results)))} SQL skeletons")

if __name__ == '__main__':
    main()

