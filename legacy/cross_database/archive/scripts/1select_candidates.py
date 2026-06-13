#!/usr/bin/env python3
"""
Select candidate SQL from single-database SQL suitable for cross-database extension.
"""

import os
import json
import argparse
import random

def load_single_database_sqls(sql_dir):
    """Load all single-database SQL."""
    all_sqls = []
    
    for db_name in sorted(os.listdir(sql_dir)):
        db_path = os.path.join(sql_dir, db_name)
        if not os.path.isdir(db_path):
            continue
        
        sql_files = [f for f in os.listdir(db_path) 
                     if f.startswith('generated_sql_') and f.endswith('.json') 
                     and not f.endswith('_error.json')]
        
        for sql_file in sql_files:
            sql_path = os.path.join(db_path, sql_file)
            try:
                with open(sql_path, 'r') as f:
                    sql_data = json.load(f)
                
                sql = sql_data.get('sql', '')
                tables = sql_data.get('tables', {})
                metadata = sql_data.get('metadata', {})
                
                # Require JOIN and at least 2 tables
                if 'JOIN' in sql.upper() and len(tables) >= 2:
                    all_sqls.append({
                        'database': db_name,
                        'file': sql_file,
                        'sql': sql,
                        'tables': list(tables.keys()),
                        'table_count': len(tables),
                        'has_join': metadata.get('has_join', False),
                        'has_subquery': metadata.get('has_subquery', False),
                        'sql_data': sql_data  # Save full data
                    })
            except Exception as e:
                print(f"Failed to read file {sql_path}: {e}")
                continue
    
    return all_sqls

def select_candidates(all_sqls, num_candidates=200, min_tables=2, max_tables=5):
    """Select candidate SQL."""
    # Filter: at least min_tables, at most max_tables
    filtered = [s for s in all_sqls 
                if min_tables <= s['table_count'] <= max_tables]
    
    # Random selection
    if len(filtered) >= num_candidates:
        selected = random.sample(filtered, num_candidates)
    else:
        selected = filtered
        # If not enough, repeat selection
        while len(selected) < num_candidates:
            selected.extend(filtered)
        selected = selected[:num_candidates]
    
    return selected

def save_candidates(candidates, output_file):
    """Save candidate SQL."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(candidates)} candidate SQL entries to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Select cross-database SQL candidates')
    parser.add_argument('--sql_dir', type=str, 
                       default='benchmark/data/beijing/output/single',
                       help='Single-database SQL directory')
    parser.add_argument('--output_file', type=str,
                       default='benchmark/generation/cross_database/candidates_2db.json',
                       help='Output file')
    parser.add_argument('--num_candidates', type=int, default=200,
                       help='Number of candidates to select (default: 200)')
    parser.add_argument('--min_tables', type=int, default=2,
                       help='Minimum number of tables (default: 2)')
    parser.add_argument('--max_tables', type=int, default=5,
                       help='Maximum number of tables (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 70)
    print("Selecting cross-database SQL candidates")
    print("=" * 70)
    print(f"\nSQL directory: {args.sql_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Number of candidates: {args.num_candidates}")
    print(f"Table count range: {args.min_tables}-{args.max_tables}")
    print()
    
    # Load SQL
    print("Loading single-database SQL...")
    all_sqls = load_single_database_sqls(args.sql_dir)
    print(f"  Found {len(all_sqls)} SQL entries containing JOIN")
    
    # Select candidates
    print(f"\nSelecting candidate SQL...")
    candidates = select_candidates(
        all_sqls, 
        args.num_candidates, 
        args.min_tables, 
        args.max_tables
    )
    print(f"  Selected {len(candidates)} candidate SQL entries")
    
    # Statistics
    table_count_dist = {}
    for sql in candidates:
        count = sql['table_count']
        table_count_dist[count] = table_count_dist.get(count, 0) + 1
    
    print(f"\nCandidate SQL table count distribution:")
    for count in sorted(table_count_dist.keys()):
        print(f"  {count} tables: {table_count_dist[count]}")
    
    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_candidates(candidates, args.output_file)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

if __name__ == '__main__':
    main()

