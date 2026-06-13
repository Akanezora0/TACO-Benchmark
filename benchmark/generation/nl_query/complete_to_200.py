#!/usr/bin/env python3
"""
Complete NL queries up to 200 entries
"""

import json
import os
import re
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import functions from the main generation script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
import importlib.util
spec = importlib.util.spec_from_file_location("generate_nl_queries", os.path.join(script_dir, "generate_nl_queries.py"))
gen_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_module)
process_single_sql = gen_module.process_single_sql

def main():
    parser = argparse.ArgumentParser(description='Complete NL queries up to 200 entries')
    parser.add_argument('--sql_dir', type=str, required=True, help='SQL file directory')
    parser.add_argument('--schema_dir', type=str, required=True, help='Schema file directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--database', type=str, required=True, help='Database name')
    parser.add_argument('--target_count', type=int, default=200, help='Target count')
    parser.add_argument('--max_workers', type=int, default=5, help='Number of concurrent worker threads')
    
    args = parser.parse_args()
    
    sql_db_dir = os.path.join(args.sql_dir, args.database)
    schema_file = os.path.join(args.schema_dir, args.database, f"{args.database}.json")
    output_db_dir = os.path.join(args.output_dir, args.database)
    
    # Get all SQL files
    sql_files = sorted([f for f in os.listdir(sql_db_dir) if f.startswith('generated_sql_') and f.endswith('.json') and '_error' not in f],
                       key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)
    
    # Get all NL query files
    nl_files = [f for f in os.listdir(output_db_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')]
    nl_indices = set()
    for f in nl_files:
        match = re.search(r'generated_nl_query_(\d+)', f)
        if match:
            nl_indices.add(int(match.group(1)))
    
    # Find missing indices
    target_indices = set(range(0, args.target_count))
    missing_indices = sorted(target_indices - nl_indices)
    
    print(f"Database: {args.database}")
    print(f"SQL file count: {len(sql_files)}")
    print(f"Current NL query count: {len(nl_indices)}")
    print(f"Missing count: {len(missing_indices)}")
    
    if not missing_indices:
        print("Target count already reached; no completion needed")
        return
    
    # Prepare tasks: generate NL queries for missing indices
    tasks = []
    sql_count = len(sql_files)
    
    for missing_idx in missing_indices:
        # Determine which SQL file and variant to use
        # By index: variant 0 uses base_idx, variant 1 uses sql_count*1+base_idx, variant 2 uses sql_count*2+base_idx
        if missing_idx < sql_count:
            # Use original SQL (variant 0)
            base_idx = missing_idx
            variant = 0
            sql_file = sql_files[base_idx]
        else:
            # Compute which variant this is
            # new_idx = sql_count * variant + base_idx
            # so: variant = (missing_idx - base_idx) // sql_count
            # but we need to find the corresponding base_idx
            # try different variants
            found = False
            for v in range(1, 4):  # variant 1, 2, 3
                base_idx = missing_idx - sql_count * v
                if 0 <= base_idx < sql_count:
                    variant = v
                    sql_file = sql_files[base_idx]
                    found = True
                    break
            
            if not found:
                # If no matching base_idx is found, use a variant of the first SQL
                base_idx = 0
                variant = (missing_idx // sql_count) + 1
                sql_file = sql_files[0]
        
        sql_file_path = os.path.join(sql_db_dir, sql_file)
        output_file = os.path.join(output_db_dir, f'generated_nl_query_{missing_idx}.json')
        
        if os.path.exists(output_file):
            continue
        
        tasks.append((sql_file_path, schema_file, output_file, variant))
    
    print(f"Preparing to generate {len(tasks)} NL queries...")
    
    # Process concurrently
    if tasks:
        total_processed = 0
        total_success = 0
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_single_sql, sql_path, schema_file, out_file, variant): (sql_path, out_file) 
                      for sql_path, _, out_file, variant in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Completing {args.database}"):
                sql_path, out_file = futures[future]
                total_processed += 1
                try:
                    if future.result():
                        total_success += 1
                except Exception as e:
                    print(f"Processing failed {sql_path}: {e}")
        
        print(f"\nDone! Processed: {total_processed}, Success: {total_success}")
        
        # Check final count again
        nl_files_final = [f for f in os.listdir(output_db_dir) if f.startswith('generated_nl_query_') and f.endswith('.json')]
        print(f"Final NL query count: {len(nl_files_final)}")

if __name__ == '__main__':
    main()
