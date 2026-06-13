#!/usr/bin/env python3
"""
Batch-fill all cross-database SQL skeletons.
"""

import os
import json
import sys
import importlib.util
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Dynamic import
sys.path.insert(0, 'benchmark/generation/cross_database')

spec = importlib.util.spec_from_file_location(
    "fill_sql",
    "benchmark/generation/cross_database/cross_db_2fill_sql_placeholders.py"
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)

load_multiple_schemas = fill_module.load_multiple_schemas
load_cross_database_graph = fill_module.load_cross_database_graph
process_cross_database_skeleton = fill_module.process_cross_database_skeleton

def process_single_skeleton(skeleton, schemas, graph_file, graph_dir, output_dir, database_dir):
    """Process a single skeleton."""
    try:
        # Process skeleton (process_cross_database_skeleton loads the graph file itself)
        # Note: process_cross_database_skeleton returns (idx, success, message) or None
        result = process_cross_database_skeleton(
            skeleton,
            schemas,
            graph_dir,  # graph_dir
            output_dir,
            database_dir
        )
        # result is (idx, success, message) or None
        if result and len(result) >= 2:
            return result[1]  # success flag
        return False
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False

def process_skeleton_file(skeleton_file_path, graph_dir, database_dir, output_dir, max_workers=5):
    """Process all skeletons in a single skeleton file."""
    import re
    
    with open(skeleton_file_path, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    
    if not skeletons:
        return 0, 0
    
    # Get involved databases
    first_skeleton = skeletons[0]
    databases = first_skeleton.get('databases', [])
    
    # Load schemas
    schemas = load_multiple_schemas(databases, database_dir)
    
    success_count = 0
    fail_count = 0
    
    # Process in parallel with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for skeleton in skeletons:
            # Find corresponding graph file
            original_file = skeleton.get('original_file', 'unknown')
            match = re.search(r'(\d+)', original_file)
            if match:
                idx = match.group(1)
                combo_name = '_'.join(sorted(databases))
                graph_file = os.path.join(graph_dir, f"cross_db_graph_{combo_name}_{idx}.json")
            else:
                import hashlib
                hash_id = hashlib.md5(skeleton['sql_skeleton'].encode()).hexdigest()[:8]
                combo_name = '_'.join(sorted(databases))
                graph_file = os.path.join(graph_dir, f"cross_db_graph_{combo_name}_{hash_id}.json")
            
            if not os.path.exists(graph_file):
                fail_count += 1
                continue
            
            future = executor.submit(
                process_single_skeleton,
                skeleton,
                schemas,
                graph_file,
                graph_dir,  # Pass graph_dir, not graph_file
                output_dir,
                database_dir
            )
            futures.append(future)
        
        # Wait for completion
        for future in as_completed(futures):
            try:
                if future.result():
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
    
    return success_count, fail_count

def main():
    import argparse
    import re
    
    parser = argparse.ArgumentParser(description='Batch-fill cross-database SQL skeletons')
    parser.add_argument('--skeleton_dir', type=str,
                       default='benchmark/generation/cross_database/skeletons',
                       help='Skeleton file directory')
    parser.add_argument('--graph_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph',
                       help='Graph file directory')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='Database directory')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single',
                       help='SQL output directory')
    parser.add_argument('--max_workers', type=int, default=5,
                       help='Maximum concurrency')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all skeleton files
    skeleton_files = []
    for f in os.listdir(args.skeleton_dir):
        if f.endswith('_skeletons.json'):
            skeleton_files.append(os.path.join(args.skeleton_dir, f))
    
    skeleton_files.sort()
    
    print("=" * 70)
    print("Batch-filling cross-database SQL skeletons")
    print("=" * 70)
    print(f"\nFound {len(skeleton_files)} skeleton files")
    
    total_success = 0
    total_fail = 0
    
    for skeleton_file in tqdm(skeleton_files, desc="Processing skeleton files"):
        success, fail = process_skeleton_file(
            skeleton_file,
            args.graph_dir,
            args.database_dir,
            args.output_dir,
            args.max_workers
        )
        total_success += success
        total_fail += fail
    
    print(f"\n" + "=" * 70)
    print(f"Done!")
    print(f"Success: {total_success}")
    print(f"Failed: {total_fail}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    main()
