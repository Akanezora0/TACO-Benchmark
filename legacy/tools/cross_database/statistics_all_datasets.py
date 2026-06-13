#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count single-database and cross-database SQL in beijing and us datasets
"""

import os
import json
from collections import defaultdict
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

def count_single_db_sqls(dataset_dir):
    """Count single-database SQL"""
    single_dir = dataset_dir / "output" / "single"
    if not single_dir.exists():
        return 0
    
    count = 0
    for db_dir in single_dir.iterdir():
        if db_dir.is_dir():
            sql_files = list(db_dir.glob("generated_sql_*.json"))
            count += len(sql_files)
    
    return count

def count_cross_db_sqls(dataset_dir, cross_db_dirs):
    """Count cross-database SQL by database count, with/without results"""
    stats_with_results = defaultdict(int)  # {2: count, 3: count, 4: count}
    stats_without_results = defaultdict(int)  # {2: count, 3: count, 4: count}
    
    for cross_db_dir_name in cross_db_dirs:
        cross_db_dir = dataset_dir / "output" / cross_db_dir_name
        if not cross_db_dir.exists():
            continue
        
        # Find all JSON files
        json_files = list(cross_db_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if has results
                results = data.get('results', [])
                has_results = results is not None and len(results) > 0
                
                # Method 1: get from metadata
                num_db = data.get('metadata', {}).get('num_databases')
                if num_db:
                    if has_results:
                        stats_with_results[num_db] += 1
                    else:
                        stats_without_results[num_db] += 1
                    continue
                
                # Method 2: get from databases field
                databases = data.get('databases', [])
                if databases:
                    num_db = len(databases)
                    if has_results:
                        stats_with_results[num_db] += 1
                    else:
                        stats_without_results[num_db] += 1
                    continue
                
                # Method 3: infer from table_database_mapping
                table_db_mapping = data.get('table_database_mapping', {})
                if table_db_mapping:
                    unique_dbs = set(table_db_mapping.values())
                    if unique_dbs:
                        num_db = len(unique_dbs)
                        if has_results:
                            stats_with_results[num_db] += 1
                        else:
                            stats_without_results[num_db] += 1
                        continue
                
                # If all fail, try inferring from filename or content
                # Default to cross-database, but cannot determine count
                # print(f"Warning: cannot determine database count for {json_file.name}")
                
            except Exception as e:
                # print(f"Error: failed to read {json_file}: {e}")
                pass
    
    return dict(stats_with_results), dict(stats_without_results)

def main():
    print("=" * 80)
    print("Dataset SQL statistics report")
    print("=" * 80)
    print()
    
    # Beijing dataset
    print("[Beijing Dataset]")
    print("-" * 80)
    beijing_dir = PROJECT_ROOT / "benchmark" / "data" / "beijing"
    
    # Single-database SQL
    single_count = count_single_db_sqls(beijing_dir)
    print(f"Single-database SQL: {single_count:,}")
    
    # Cross-database SQL
    cross_db_dirs = [
        "cross_db_single_join",
        "cross_db_single_join_backup_51",
        "cross_db_single",
        "cross_db_final"
    ]
    
    cross_db_stats_with, cross_db_stats_without = count_cross_db_sqls(beijing_dir, cross_db_dirs)
    
    total_cross_with = sum(cross_db_stats_with.values())
    total_cross_without = sum(cross_db_stats_without.values())
    total_cross = total_cross_with + total_cross_without
    
    print(f"Total cross-database SQL: {total_cross:,}")
    print(f"  with results: {total_cross_with:,}")
    print(f"  without results: {total_cross_without:,}")
    
    if cross_db_stats_with or cross_db_stats_without:
        print("\nDistribution by database count:")
        all_db_counts = set(list(cross_db_stats_with.keys()) + list(cross_db_stats_without.keys()))
        for num_db in sorted(all_db_counts):
            count_with = cross_db_stats_with.get(num_db, 0)
            count_without = cross_db_stats_without.get(num_db, 0)
            count_total = count_with + count_without
            if count_total > 0:
                print(f"  Cross {num_db} databases: {count_total:,} (with results: {count_with:,}, without results: {count_without:,})")
    else:
        print("  (No cross-database SQL found)")
    
    print()
    
    # US dataset
    print("[US Dataset]")
    print("-" * 80)
    us_dir = PROJECT_ROOT / "benchmark" / "data" / "us"
    
    # Single-database SQL
    single_count_us = count_single_db_sqls(us_dir)
    print(f"Single-database SQL: {single_count_us:,}")
    
    # Cross-database SQL
    cross_db_stats_us_with, cross_db_stats_us_without = count_cross_db_sqls(us_dir, cross_db_dirs)
    
    total_cross_us_with = sum(cross_db_stats_us_with.values())
    total_cross_us_without = sum(cross_db_stats_us_without.values())
    total_cross_us = total_cross_us_with + total_cross_us_without
    
    print(f"Total cross-database SQL: {total_cross_us:,}")
    if total_cross_us > 0:
        print(f"  with results: {total_cross_us_with:,}")
        print(f"  without results: {total_cross_us_without:,}")
    
    if cross_db_stats_us_with or cross_db_stats_us_without:
        print("\nDistribution by database count:")
        all_db_counts_us = set(list(cross_db_stats_us_with.keys()) + list(cross_db_stats_us_without.keys()))
        for num_db in sorted(all_db_counts_us):
            count_with = cross_db_stats_us_with.get(num_db, 0)
            count_without = cross_db_stats_us_without.get(num_db, 0)
            count_total = count_with + count_without
            if count_total > 0:
                print(f"  Cross {num_db} databases: {count_total:,} (with results: {count_with:,}, without results: {count_without:,})")
    else:
        print("  (No cross-database SQL found)")
    
    print()
    
    # Total
    print("=" * 80)
    print("[Total]")
    print("-" * 80)
    print(f"Total single-database SQL: {single_count + single_count_us:,}")
    print(f"  Beijing: {single_count:,}")
    print(f"  US: {single_count_us:,}")
    print()
    print(f"Total cross-database SQL: {total_cross + total_cross_us:,}")
    print(f"  Beijing: {total_cross:,} (with results: {total_cross_with:,}, without results: {total_cross_without:,})")
    print(f"  US: {total_cross_us:,} (with results: {total_cross_us_with:,}, without results: {total_cross_us_without:,})")
    print()
    
    # Combined statistics (with-results counts only)
    all_cross_stats_with = defaultdict(int)
    all_cross_stats_without = defaultdict(int)
    
    for num_db, count in cross_db_stats_with.items():
        all_cross_stats_with[num_db] += count
    for num_db, count in cross_db_stats_without.items():
        all_cross_stats_without[num_db] += count
    for num_db, count in cross_db_stats_us_with.items():
        all_cross_stats_with[num_db] += count
    for num_db, count in cross_db_stats_us_without.items():
        all_cross_stats_without[num_db] += count
    
    if all_cross_stats_with or all_cross_stats_without:
        print("Cross-database SQL distribution by count (merged, with results only):")
        all_db_counts = set(list(all_cross_stats_with.keys()) + list(all_cross_stats_without.keys()))
        for num_db in sorted(all_db_counts):
            count_with = all_cross_stats_with.get(num_db, 0)
            count_without = all_cross_stats_without.get(num_db, 0)
            count_total = count_with + count_without
            if count_total > 0:
                print(f"  Cross {num_db} databases: {count_with:,} (total: {count_total:,})")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
