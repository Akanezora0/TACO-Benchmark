#!/usr/bin/env python3
"""
Optimized SQL-Schema Linking Graph construction script

Optimizations:
1. Do not save full GraphML files; only save compact JSON metadata
2. Metadata contains only essential table, column, and foreign key information
3. Significantly reduce file size for use in prompts
"""

import json
import os
from tqdm import tqdm

def load_schema(schema_file):
    """Load database schema information"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if standard schema format (contains 'tables' key)
    if 'tables' in data:
        return data
    
    # If not standard format, extract schema from database JSON file
    schema = {'tables': []}
    for table_name, table_data in data.items():
        columns = []
        if 'columns' in table_data:
            for col_name in table_data['columns']:
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'
                })
        
        schema['tables'].append({
            'table_name': table_name,
            'table_comment': table_name,
            'table_description': 'No description available.',
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        })
    
    return schema

def extract_schema_metadata(schema_info):
    """
    Extract compact metadata from schema.
    Contains only essential table, column, and foreign key information.
    """
    metadata = {
        'tables': {},
        'foreign_keys': []
    }
    
    # Extract table information
    for table in schema_info['tables']:
        table_name = table['table_name']
        columns = []
        for column in table['columns']:
            columns.append({
                'name': column['column_name'],
                'data_type': column.get('data_type', 'TEXT')
            })
        
        metadata['tables'][table_name] = {
            'name': table_name,
            'comment': table.get('table_comment', ''),
            'description': table.get('table_description', 'No description available.'),
            'columns': columns
        }
    
    # Extract foreign key relations
    for table in schema_info['tables']:
        for fk in table.get('foreign_keys', []):
            metadata['foreign_keys'].append({
                'source_table': fk.get('table', table['table_name']),
                'source_column': fk.get('column', ''),
                'target_table': fk.get('references', {}).get('table', ''),
                'target_column': fk.get('references', {}).get('column', '')
            })
    
    return metadata

def save_metadata(metadata, output_file):
    """Save compact metadata as JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def process_database(database_name, skeleton_file, schema_file, output_dir):
    """
    Process a single database and generate compact metadata files.
    """
    # Load schema
    if not os.path.exists(schema_file):
        print(f"Schema file does not exist: {schema_file}")
        return
    
    schema = load_schema(schema_file)
    
    # Load SQL skeletons
    if not os.path.exists(skeleton_file):
        print(f"SQL skeleton file does not exist: {skeleton_file}")
        return
    
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        sql_skeletons = json.load(f)
    
    # Create output directory
    graph_dir = os.path.join(output_dir, database_name)
    os.makedirs(graph_dir, exist_ok=True)
    
    print(f"Processing database '{database_name}', {len(sql_skeletons)} SQL skeletons...")
    
    # Extract schema metadata (shared by all SQL skeletons)
    schema_metadata = extract_schema_metadata(schema)
    
    # Save metadata for each SQL skeleton (shared schema, but saved per skeleton for compatibility)
    for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} progress", leave=False)):
        # If dict, extract sql_framework field
        if isinstance(sql_skeleton, dict):
            sql_framework = sql_skeleton.get('sql_framework', '')
        else:
            sql_framework = sql_skeleton
        
        if not sql_framework:
            continue
        
        # Create metadata with SQL skeleton information
        metadata = {
            'sql_framework': sql_framework,
            'database_name': database_name,
            'skeleton_index': idx,
            **schema_metadata
        }
        
        # Save metadata (compact format, no graph structure)
        metadata_file = os.path.join(graph_dir, f"{database_name}_metadata_{idx}.json")
        save_metadata(metadata, metadata_file)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build SQL-Schema metadata (optimized version)')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser.add_argument('--skeleton_dir', type=str, default=None,
                       help='SQL skeleton directory (default: ../../data/beijing/output/sql_skeleton)')
    parser.add_argument('--database_dir', type=str, default=None,
                       help='Database directory (default: ../../data/beijing/database)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../../data/beijing/output/graph)')
    
    args = parser.parse_args()
    
    # Set default paths
    if args.skeleton_dir is None:
        args.skeleton_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'sql_skeleton')
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'graph')
    
    # Convert to absolute paths
    args.skeleton_dir = os.path.abspath(args.skeleton_dir)
    args.database_dir = os.path.abspath(args.database_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all SQL skeleton files
    skeleton_files = [f for f in os.listdir(args.skeleton_dir) if f.endswith('_sql_skeleton.json')]
    
    print(f"Found {len(skeleton_files)} database SQL skeleton files")
    
    for skeleton_file in tqdm(skeleton_files, desc="Overall progress"):
        # Extract database name
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        
        skeleton_path = os.path.join(args.skeleton_dir, skeleton_file)
        schema_path = os.path.join(args.database_dir, database_name, f"{database_name}.json")
        
        process_database(database_name, skeleton_path, schema_path, args.output_dir)
    
    print(f"\n{'='*60}")
    print("✓ Metadata construction complete for all databases!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
