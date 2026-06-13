#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check US dataset cross-database SQL generation status
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Target counts
TARGET_COUNTS = {
    2: 900,  # cross 2 databases
    3: 264,  # cross 3 databases
    4: 6     # cross 4 databases
}

# Default path
DEFAULT_SQL_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output" / "cross_db_single_join"

def count_sqls_by_db_count(sql_dir):
    """Count SQL files by database count (only those with results)"""
    stats = defaultdict(int)  # {2: count, 3: count, 4: count}
    total = 0
    with_results = 0
    
    if not sql_dir.exists():
        return stats, total, with_results
    
    for sql_file in sql_dir.glob("cross_db_generated_sql_*.json"):
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Get database count
                num_db = data.get('metadata', {}).get('num_databases')
                if not num_db:
                    databases = data.get('databases', [])
                    if databases:
                        num_db = len(databases)
                    else:
                        table_db_mapping = data.get('table_database_mapping', {})
                        if table_db_mapping:
                            unique_dbs = set(table_db_mapping.values())
                            num_db = len(unique_dbs) if unique_dbs else 0
                
                if num_db in [2, 3, 4]:
                    total += 1
                    results = data.get('results', [])
                    if results is not None and len(results) > 0:
                        stats[num_db] += 1
                        with_results += 1
        except:
            pass
    
    return stats, total, with_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check US dataset cross-database SQL generation status')
    parser.add_argument('--sql-dir', type=str, default=None,
                       help=f'SQL file directory (default: {DEFAULT_SQL_DIR})')
    
    args = parser.parse_args()
    
    sql_dir = Path(args.sql_dir) if args.sql_dir else DEFAULT_SQL_DIR
    
    print("=" * 80)
    print("US dataset cross-database SQL generation status")
    print("=" * 80)
    print()
    
    stats, total, with_results = count_sqls_by_db_count(sql_dir)
    
    print(f"Total SQL files: {total}")
    print(f"SQL with results: {with_results} ({with_results/total*100:.1f}%)" if total > 0 else "SQL with results: 0")
    print()
    
    print("Distribution by database count:")
    print("-" * 80)
    print(f"{'DB count':<15} {'With results':<10} {'Target':<10} {'Remaining':<10} {'Progress':<10} {'Status':<10}")
    print("-" * 80)
    
    total_needed = 0
    total_with_results = 0
    total_target = 0
    
    for db_count in sorted([2, 3, 4]):
        with_result = stats.get(db_count, 0)
        target = TARGET_COUNTS[db_count]
        needed = max(0, target - with_result)
        
        total_with_results += with_result
        total_target += target
        total_needed += needed
        
        completion = (with_result / target * 100) if target > 0 else 0
        status = "✅ Done" if with_result >= target else "⏳ In progress"
        
        print(f"{db_count} databases   {with_result:<10} {target:<10} {needed:<10} {completion:>6.1f}%    {status:<10}")
    
    print("-" * 80)
    print(f"{'Total':<15} {total_with_results:<10} {total_target:<10} {total_needed:<10} {(total_with_results/total_target*100) if total_target > 0 else 0:>6.1f}%")
    print("=" * 80)
    
    if total_needed > 0:
        print(f"\nStill need to generate: {total_needed} SQL files with results")
        print("Suggested command: python3 run_all.py or run generation scripts step by step")

if __name__ == '__main__':
    main()
