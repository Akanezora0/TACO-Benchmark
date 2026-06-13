#!/usr/bin/env python3
"""
Test SQL generation for 3-database and 4-database scenarios.
Generate a small batch first to verify the pipeline works.
"""

import os
import json
import argparse
from tqdm import tqdm
import sys

# Import SQL filling module
sys.path.insert(0, os.path.dirname(__file__))
from cross_db_2fill_sql_placeholders_join import (
    process_cross_database_skeleton
)

# Import load_schema function
import importlib.util
sql_filling_dir = os.path.join(os.path.dirname(__file__), '..', 'sql_filling')
spec = importlib.util.spec_from_file_location(
    "fill_sql_placeholders_improved",
    os.path.join(sql_filling_dir, "2fill_sql_placeholders_improved.py")
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)
load_schema = fill_module.load_schema

def test_3db_4db_generation(skeleton_file, graph_dir, database_dir, output_dir, 
                            num_3db=5, num_4db=3):
    """Test generation of SQL spanning 3 and 4 databases."""
    
    # Load skeletons
    print("Loading SQL skeletons...")
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        all_skeletons = json.load(f)
    
    # Filter skeletons for 3 and 4 databases
    skeletons_3db = [s for s in all_skeletons if s.get('num_databases') == 3]
    skeletons_4db = [s for s in all_skeletons if s.get('num_databases') == 4]
    
    print(f"  Found {len(skeletons_3db)} 3-database skeletons")
    print(f"  Found {len(skeletons_4db)} 4-database skeletons")
    
    # Select skeletons to test
    test_skeletons = skeletons_3db[:num_3db] + skeletons_4db[:num_4db]
    print(f"\nWill test {len(test_skeletons)} skeletons ({num_3db} with 3 databases + {num_4db} with 4 databases)")
    
    # Collect all involved databases
    all_databases = set()
    for skeleton in test_skeletons:
        all_databases.update(skeleton.get('databases', []))
    
    print(f"Involved databases: {sorted(all_databases)}")
    
    # Load schemas for all databases
    print("\nLoading database schemas...")
    # Expected path format for load_multiple_schemas: database_dir/db_name/db_name.json
    schemas = {}
    for db_name in all_databases:
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            try:
                schema = load_schema(schema_file)
                if schema:
                    schemas[db_name] = schema
            except Exception as e:
                print(f"  Warning: failed to load schema {schema_file}: {e}")
        else:
            print(f"  Warning: schema file not found {schema_file}")
    print(f"Successfully loaded schemas for {len(schemas)} databases")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each skeleton
    print("\nStarting SQL generation...")
    success_count = 0
    has_results_count = 0
    failed_count = 0
    
    for skeleton in tqdm(test_skeletons, desc="Generating SQL"):
        # Find corresponding graph file
        original_file = skeleton.get('original_file', '')
        match = __import__('re').search(r'(\d+)', original_file)
        if match:
            graph_idx = match.group(1)
            graph_file = os.path.join(graph_dir, f"cross_db_graph_{graph_idx}.json")
        else:
            graph_file = None
        
        if not graph_file or not os.path.exists(graph_file):
            print(f"  Warning: graph file not found {graph_file}")
            failed_count += 1
            continue
        
        # Load graph file
        with open(graph_file, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Process skeleton
        try:
            # process_cross_database_skeleton expects graph_dir, not graph_data
            # We need to find graph_dir
            graph_dir = os.path.dirname(graph_file)
            
            # Call processing function
            # process_cross_database_skeleton returns: (idx, success, message)
            idx, success, message = process_cross_database_skeleton(
                skeleton,
                schemas,
                graph_dir,
                output_dir,
                database_dir
            )
            
            # Build output file path from returned idx
            result_file = os.path.join(output_dir, f"cross_db_generated_sql_{idx}.json")
            
            # Check results
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                results = result_data.get('results', [])
                if results is not None and len(results) > 0:
                    has_results_count += 1
                    success_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed_count += 1
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Test result summary")
    print("=" * 70)
    print(f"Total tests: {len(test_skeletons)}")
    print(f"Generated successfully: {success_count} ({success_count/len(test_skeletons)*100:.1f}%)")
    print(f"With results: {has_results_count} ({has_results_count/len(test_skeletons)*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/len(test_skeletons)*100:.1f}%)")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='Test SQL generation for 3-database and 4-database scenarios')
    parser.add_argument('--skeleton_file', type=str,
                       default='benchmark/generation/cross_database/cross_db_skeletons_join.json',
                       help='SQL skeleton file')
    parser.add_argument('--graph_dir', type=str,
                       default='benchmark/generation/cross_database/cross_db_graphs_join',
                       help='Graph file directory')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='Database directory')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='Output directory')
    parser.add_argument('--num_3db', type=int, default=5,
                       help='Number of 3-database skeletons to test')
    parser.add_argument('--num_4db', type=int, default=3,
                       help='Number of 4-database skeletons to test')
    
    args = parser.parse_args()
    
    test_3db_4db_generation(
        args.skeleton_file,
        args.graph_dir,
        args.database_dir,
        args.output_dir,
        args.num_3db,
        args.num_4db
    )

if __name__ == '__main__':
    main()
