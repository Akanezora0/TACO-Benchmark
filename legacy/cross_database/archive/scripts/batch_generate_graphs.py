#!/usr/bin/env python3
"""
Batch-generate graph files for all cross-database SQL skeletons.
"""

import os
import json
import sys
import importlib.util
from pathlib import Path
from tqdm import tqdm

# Dynamically import cross-database graph generation module
sys.path.insert(0, 'benchmark/generation/cross_database')

spec = importlib.util.spec_from_file_location(
    "cross_db_graphs",
    "benchmark/generation/cross_database/cross_db_1build_schema_graphs.py"
)
graph_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_module)

load_multiple_schemas = graph_module.load_multiple_schemas
process_cross_database_skeleton = graph_module.process_cross_database_skeleton

def process_all_skeleton_files(skeleton_dir, database_dir, output_dir):
    """Process all skeleton files and generate graph files."""
    
    # Get all skeleton files
    skeleton_files = []
    for f in os.listdir(skeleton_dir):
        if f.endswith('_skeletons.json'):
            skeleton_files.append(f)
    
    skeleton_files.sort()
    
    print("=" * 70)
    print("Batch-generating cross-database graph files")
    print("=" * 70)
    print(f"\nFound {len(skeleton_files)} skeleton files")
    
    total_skeletons = 0
    total_graphs = 0
    
    # Process by combination (same-combination skeleton files share schemas)
    combo_schemas_cache = {}  # Cache loaded schemas
    
    for skeleton_file in tqdm(skeleton_files, desc="Processing skeleton files"):
        skeleton_path = os.path.join(skeleton_dir, skeleton_file)
        
        try:
            with open(skeleton_path, 'r', encoding='utf-8') as f:
                skeletons = json.load(f)
            
            if not skeletons:
                continue
            
            # Get involved databases (from first skeleton; all skeletons in file share the same databases)
            first_skeleton = skeletons[0]
            databases = first_skeleton.get('databases', [])
            combo_key = tuple(sorted(databases))
            
            # Load schemas (if not already loaded)
            if combo_key not in combo_schemas_cache:
                schemas = load_multiple_schemas(databases, database_dir)
                combo_schemas_cache[combo_key] = schemas
                print(f"\nLoaded schemas for database combination {combo_key}: {len(schemas)} databases")
            else:
                schemas = combo_schemas_cache[combo_key]
            
            # Generate graph for each skeleton
            for skeleton in skeletons:
                try:
                    # Generate graph (process_cross_database_skeleton handles filenames automatically)
                    output_file = process_cross_database_skeleton(skeleton, schemas, output_dir)
                    if output_file:
                        total_graphs += 1
                        total_skeletons += 1
                    
                except Exception as e:
                    print(f"\nFailed to process skeleton {skeleton_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
        except Exception as e:
            print(f"\nFailed to read skeleton file {skeleton_file}: {e}")
            continue
    
    print(f"\n" + "=" * 70)
    print(f"Done!")
    print(f"Processed {len(skeleton_files)} skeleton files")
    print(f"Generated {total_graphs} graph files")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

if __name__ == '__main__':
    import argparse
    import re
    
    parser = argparse.ArgumentParser(description='Batch-generate cross-database graph files')
    parser.add_argument('--skeleton_dir', type=str,
                       default='benchmark/generation/cross_database/skeletons',
                       help='Skeleton file directory')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='Database directory')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph',
                       help='Graph file output directory')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    process_all_skeleton_files(args.skeleton_dir, args.database_dir, args.output_dir)
