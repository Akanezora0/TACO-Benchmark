#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check SQL and NL query generation status for the US dataset
"""

import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output"
SINGLE_DIR = OUTPUT_DIR / "single"
NL_DIR = OUTPUT_DIR / "nl_query"

def count_sqls(db_name):
    """Count SQLs (only those with results)"""
    sql_path = SINGLE_DIR / db_name
    if not sql_path.exists():
        return 0
    
    count = 0
    for sql_file in sql_path.glob("generated_sql_*.json"):
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Only count SQLs with results
                if 'results' in data and data['results'] is not None:
                    count += 1
        except:
            pass
    
    return count

def count_nl_queries(db_name):
    """Count NL queries"""
    nl_path = NL_DIR / db_name
    if not nl_path.exists():
        return 0
    
    count = 0
    for nl_file in nl_path.glob("*.json"):
        try:
            with open(nl_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check for natural_language_query field
                if 'natural_language_query' in data and data['natural_language_query']:
                    count += 1
        except:
            pass
    
    return count

def main():
    print("US Dataset SQL and NL Query Generation Status")
    print("=" * 100)
    
    # Get all databases
    databases = []
    if SINGLE_DIR.exists():
        databases = sorted([d.name for d in SINGLE_DIR.iterdir() if d.is_dir()])
    
    if not databases:
        print("No databases found")
        return
    
    print(f"\n{'Database Name':<50} {'SQL Count':<10} {'NL Count':<12} {'Status':<20}")
    print("-" * 100)
    
    total_sql = 0
    total_nl = 0
    completed_sql = 0
    completed_nl = 0
    target_count = 220  # Target count
    
    for db_name in databases:
        sql_count = count_sqls(db_name)
        nl_count = count_nl_queries(db_name)
        
        total_sql += sql_count
        total_nl += nl_count
        
        # Determine status
        if sql_count >= target_count and nl_count >= target_count:
            status = "✅ Complete (SQL+NL)"
            completed_sql += 1
            completed_nl += 1
        elif sql_count >= target_count:
            status = f"⏳ SQL only (missing {target_count - nl_count} NL)"
            completed_sql += 1
        elif nl_count >= target_count:
            status = f"⚠️  NL only (missing {target_count - sql_count} SQL)"
            completed_nl += 1
        else:
            status = f"❌ Incomplete (SQL missing {target_count - sql_count}, NL missing {target_count - nl_count})"
        
        display_name = db_name[:47] + "..." if len(db_name) > 50 else db_name
        print(f"{display_name:<50} {sql_count:<10} {nl_count:<12} {status:<20}")
    
    print("-" * 100)
    print(f"{'Total':<50} {total_sql:<10} {total_nl:<12} {completed_sql}/{len(databases)} SQL done, {completed_nl}/{len(databases)} NL done")
    print("=" * 100)
    
    # Summary statistics
    print("\nSummary:")
    print(f"  - Total databases: {len(databases)}")
    print(f"  - Total SQLs: {total_sql} (target: {len(databases) * target_count})")
    print(f"  - Total NL queries: {total_nl} (target: {len(databases) * target_count})")
    print(f"  - SQL completion: {completed_sql}/{len(databases)} ({completed_sql/len(databases)*100:.1f}%)")
    print(f"  - NL completion: {completed_nl}/{len(databases)} ({completed_nl/len(databases)*100:.1f}%)")
    
    # Find databases that need NL query generation
    need_nl = []
    for db_name in databases:
        sql_count = count_sqls(db_name)
        nl_count = count_nl_queries(db_name)
        if sql_count >= target_count and nl_count < target_count:
            need_nl.append((db_name, sql_count, nl_count, target_count - nl_count))
    
    if need_nl:
        print(f"\nDatabases needing NL query generation ({len(need_nl)}):")
        for db_name, sql_count, nl_count, need in need_nl:
            print(f"  - {db_name}: SQL {sql_count}, NL {nl_count}, need {need} more NL queries")

if __name__ == '__main__':
    main()
