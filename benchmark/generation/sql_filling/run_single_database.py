#!/usr/bin/env python3
"""
SQL filling script for a single database only.
Used to complete SQL skeleton generation for a specific database.
"""

import sys
import os

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import importlib.util
import argparse

# Dynamic import (module name starts with a digit)
spec = importlib.util.spec_from_file_location(
    "fill_module", 
    os.path.join(script_dir, "fill_sql_placeholders.py")
)
fill_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill_module)

process_single_database = fill_module.process_single_database
load_config = fill_module.load_config

def main():
    parser = argparse.ArgumentParser(description='Process SQL filling for a single database')
    parser.add_argument('--database_name', type=str, required=True,
                       help='Database name (e.g., 医疗健康)')
    parser.add_argument('--database_dir', type=str, 
                       default='../../data/beijing/database_chinese',
                       help='Database directory')
    parser.add_argument('--skeleton_dir', type=str,
                       default='../../data/beijing/output/sql_skeleton',
                       help='SQL skeleton directory')
    parser.add_argument('--graph_dir', type=str,
                       default='../../data/beijing/output/graph_chinese',
                       help='Graph file directory')
    parser.add_argument('--output_dir', type=str,
                       default='../../data/beijing/output',
                       help='Output directory')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='Maximum retry times (default: 3)')
    parser.add_argument('--config', type=str, default=None,
                       help='Config file path (default: ./config.yaml)')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    database_dir = os.path.abspath(os.path.join(script_dir, args.database_dir))
    skeleton_dir = os.path.abspath(os.path.join(script_dir, args.skeleton_dir))
    graph_dir = os.path.abspath(os.path.join(script_dir, args.graph_dir))
    output_dir = os.path.abspath(os.path.join(script_dir, args.output_dir))
    
    # Build file paths
    skeleton_file = os.path.join(skeleton_dir, f"{args.database_name}_sql_skeleton.json")
    schema_file = os.path.join(database_dir, args.database_name, f"{args.database_name}.json")
    
    # Check if files exist
    if not os.path.exists(skeleton_file):
        print(f"Error: SQL skeleton file does not exist: {skeleton_file}")
        return
    
    if not os.path.exists(schema_file):
        print(f"Error: Schema file does not exist: {schema_file}")
        return
    
    # Load config
    config_file = args.config if args.config else os.path.join(script_dir, 'config.yaml')
    fill_module.API_CONFIG = fill_module.load_config(config_file)
    
    print(f"=== Processing Database: {args.database_name} ===")
    print(f"SQL skeleton file: {skeleton_file}")
    print(f"Schema file: {schema_file}")
    print(f"Graph file directory: {graph_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Maximum retries: {args.max_retries}")
    print()
    
    # Process database
    success_count, fail_count = process_single_database(
        args.database_name,
        skeleton_file,
        schema_file,
        graph_dir,
        output_dir,
        max_retries=args.max_retries
    )
    
    print()
    print(f"=== Processing Complete ===")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total: {success_count + fail_count}")

if __name__ == '__main__':
    main()
