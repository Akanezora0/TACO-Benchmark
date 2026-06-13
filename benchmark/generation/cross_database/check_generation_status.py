#!/usr/bin/env python3
"""
Count SQL files with results

Statistics by 2, 3, and 4 database categories, with target counts
"""

import os
import json
import argparse
from collections import defaultdict

# Target counts (from paper statistics)
TARGET_COUNTS = {
    2: 359,  # cross 2 databases
    3: 105,  # cross 3 databases
    4: 2     # cross 4 databases
}

def check_status(sql_dir):
    """Count SQL files with results"""
    
    stats_by_db_count = defaultdict(lambda: {'total': 0, 'with_results': 0, 'without_results': 0})
    all_files = []
    
    print("Counting SQL files...")
    for f in sorted(os.listdir(sql_dir)):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    num_databases = data.get('metadata', {}).get('num_databases', 
                                                                    len(data.get('databases', [])))
                    
                    stats_by_db_count[num_databases]['total'] += 1
                    
                    if results is not None and len(results) > 0:
                        stats_by_db_count[num_databases]['with_results'] += 1
                        all_files.append((num_databases, f))
                    else:
                        stats_by_db_count[num_databases]['without_results'] += 1
            except Exception as e:
                print(f"Error processing file {f}: {e}")
    
    # Display statistics
    print("\n" + "=" * 70)
    print("SQL generation status statistics")
    print("=" * 70)
    
    total_with_results = 0
    total_without_results = 0
    total_target = 0
    
    for db_count in sorted(stats_by_db_count.keys()):
        stats = stats_by_db_count[db_count]
        target = TARGET_COUNTS.get(db_count, 0)
        total_target += target
        
        with_results = stats['with_results']
        without_results = stats['without_results']
        total = stats['total']
        
        total_with_results += with_results
        total_without_results += without_results
        
        progress = (with_results / target * 100) if target > 0 else 0
        
        print(f"\nCross {db_count} databases:")
        print(f"  With results: {with_results} / {target} ({progress:.1f}%)")
        print(f"  Without results: {without_results}")
        print(f"  Total: {total}")
        
        if with_results < target:
            print(f"  ⚠️  Still need: {target - with_results}")
        else:
            print(f"  ✅ Target reached")
    
    print(f"\nTotal:")
    print(f"  With results: {total_with_results} / {total_target} ({total_with_results/total_target*100:.1f}%)")
    print(f"  Without results: {total_without_results}")
    print(f"  Total: {total_with_results + total_without_results}")
    
    # Show file index range
    if all_files:
        print(f"\nFile index range:")
        for db_count in sorted(set(f[0] for f in all_files)):
            files = [f[1] for f in all_files if f[0] == db_count]
            indices = []
            for f in files:
                try:
                    idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                    indices.append(idx)
                except:
                    pass
            
            if indices:
                print(f"  {db_count} databases: {min(indices)} - {max(indices)} ({len(indices)} files)")
    
    print("\n" + "=" * 70)
    
    return stats_by_db_count

def main():
    parser = argparse.ArgumentParser(description='Count SQL files with results')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL file directory')
    
    args = parser.parse_args()
    
    check_status(args.sql_dir)

if __name__ == '__main__':
    main()

