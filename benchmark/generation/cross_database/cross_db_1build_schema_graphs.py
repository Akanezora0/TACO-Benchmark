#!/usr/bin/env python3
"""
Cross-database SQL-Schema Linking Graph construction script.

Extends the single-database graph generation script to support cross-database scenarios:
1. Load schemas from multiple databases
2. Generate graphs for cross-database SQL skeletons
3. Consider cross-database table and column relationships
"""

import json
import networkx as nx
import re
import os
import importlib.util
from tqdm import tqdm
from collections import defaultdict
import argparse

# Import single-database graph generation functions
import sys
sql_filling_dir = os.path.join(os.path.dirname(__file__), '..', 'sql_filling')
sys.path.insert(0, sql_filling_dir)

# Dynamic import
import importlib.util
spec = importlib.util.spec_from_file_location(
    "build_schema_graphs_improved",
    os.path.join(sql_filling_dir, "build_schema_graphs.py")
)
build_graphs_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_graphs_module)

load_schema = build_graphs_module.load_schema
parse_sql_framework = build_graphs_module.parse_sql_framework
get_possible_schema_nodes = build_graphs_module.get_possible_schema_nodes

def load_multiple_schemas(database_names, database_dir):
    """Load schema information from multiple databases."""
    schemas = {}
    for db_name in database_names:
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            schemas[db_name] = load_schema(schema_file)
        else:
            print(f"Warning: schema file not found {schema_file}")
    return schemas

def build_cross_database_graph(sql_skeleton, schemas, table_database_mapping):
    """
    Build a graph for a cross-database SQL skeleton.
    Key point: load schemas from all involved databases and construct a unified graph.

    Args:
        sql_skeleton: SQL skeleton string
        schemas: Dictionary of schemas from multiple databases {db_name: schema}
        table_database_mapping: Mapping from table to database {table_name: db_name}

    Returns:
        NetworkX graph object and metadata
    """
    G = nx.DiGraph()

    # 1. Add table and column nodes from all involved databases (unified construction)
    # Key: build one graph for all databases rather than separate graphs per database
    for db_name, schema in schemas.items():
        if schema is None:
            continue

        for table_info in schema.get('tables', []):
            table_name = table_info.get('table_name', '')
            # Node ID format: database_name.table_name (ensures cross-database uniqueness)
            table_node_id = f"{db_name}.{table_name}"

            # Add table node
            G.add_node(table_node_id,
                      node_type='table',
                      table_name=table_name,
                      database=db_name,
                      table_comment=table_info.get('table_comment', ''),
                      table_description=table_info.get('table_description', ''))

            # Add column nodes
            for col_info in table_info.get('columns', []):
                col_name = col_info.get('column_name', '')
                # Node ID format: database_name.table_name.column_name
                col_node_id = f"{db_name}.{table_name}.{col_name}"

                G.add_node(col_node_id,
                          node_type='column',
                          column_name=col_name,
                          table_name=table_name,
                          database=db_name,
                          data_type=col_info.get('data_type', 'TEXT'))

                # Add table-to-column edge
                G.add_edge(table_node_id, col_node_id, edge_type='contains')

    # 2. Add foreign key relationships (including cross-database foreign keys)
    for db_name, schema in schemas.items():
        for table_info in schema.get('tables', []):
            table_name = table_info.get('table_name', '')
            table_node_id = f"{db_name}.{table_name}"

            # Process foreign key relationships
            for fk in table_info.get('foreign_keys', []):
                ref_table = fk.get('referenced_table', '')
                ref_col = fk.get('referenced_column', '')
                fk_col = fk.get('column_name', '')

                # Find the database containing the referenced table
                ref_db = table_database_mapping.get(ref_table, db_name)  # Default to same database

                if ref_table and ref_col:
                    source_col_id = f"{db_name}.{table_name}.{fk_col}"
                    target_col_id = f"{ref_db}.{ref_table}.{ref_col}"

                    # Add foreign key edge if the target column node exists
                    if G.has_node(target_col_id):
                        G.add_edge(source_col_id, target_col_id, edge_type='foreign_key')

    # 3. Parse SQL skeleton and identify placeholders
    placeholders = parse_sql_framework(sql_skeleton)

    # 4. Add possible connections for each placeholder
    placeholder_nodes = {}
    for i, placeholder in enumerate(placeholders):
        placeholder_id = f"placeholder_{i}"
        placeholder_nodes[placeholder_id] = placeholder

        # Get possible schema nodes based on placeholder type
        # Note: cross-database cases must be considered here
        possible_nodes = get_possible_schema_nodes(G, placeholder['clause'])

        # Filter: only consider tables in table_database_mapping
        relevant_tables = set(table_database_mapping.keys())
        filtered_nodes = []
        for node_id in possible_nodes:
            # Extract table name from node ID
            if '.' in node_id:
                parts = node_id.split('.')
                if len(parts) >= 2:
                    table_name = parts[1]  # Format: db.table or db.table.col
                    if table_name in relevant_tables:
                        filtered_nodes.append(node_id)

        # Add placeholder node
        G.add_node(placeholder_id,
                  node_type='placeholder',
                  placeholder_type=placeholder['clause'],
                  position=placeholder['position'])

        # Connect to possible schema nodes
        for node_id in filtered_nodes:
            G.add_edge(placeholder_id, node_id, edge_type='possible_match')

    # 5. Build metadata
    metadata = {
        'databases': list(schemas.keys()),
        'table_database_mapping': table_database_mapping,
        'placeholders': placeholders,
        'num_tables': len(set(table_database_mapping.keys())),
        'num_databases': len(schemas)
    }

    return G, metadata

def process_cross_database_skeleton(skeleton_data, schemas, output_dir):
    """Process a single cross-database SQL skeleton and generate a graph file."""
    sql_skeleton = skeleton_data['sql_skeleton']
    table_database_mapping = skeleton_data['table_database_mapping']

    # Build graph
    G, metadata = build_cross_database_graph(
        sql_skeleton,
        schemas,
        table_database_mapping
    )

    # Save graph (using NetworkX JSON format)
    graph_data = {
        'nodes': [
            {
                'id': node_id,
                **attr
            }
            for node_id, attr in G.nodes(data=True)
        ],
        'edges': [
            {
                'source': source,
                'target': target,
                **attr
            }
            for source, target, attr in G.edges(data=True)
        ],
        'metadata': metadata
    }

    # Determine output filename (includes combination info)
    original_file = skeleton_data.get('original_file', 'unknown')
    databases = skeleton_data.get('databases', [])
    combo_name = '_'.join(sorted(databases)) if databases else 'unknown'

    # Extract index from original_file (e.g. generated_sql_1.json -> 1 or join_skeleton_0 -> 0)
    match = re.search(r'(\d+)', original_file)
    if match:
        idx = match.group(1)
        output_file = os.path.join(output_dir, f"cross_db_graph_{idx}.json")
    else:
        # Use hash if no index is found
        import hashlib
        hash_id = hashlib.md5(sql_skeleton.encode()).hexdigest()[:8]
        output_file = os.path.join(output_dir, f"cross_db_graph_{hash_id}.json")

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    return output_file

def main():
    parser = argparse.ArgumentParser(description='Generate graphs for cross-database SQL skeletons')
    parser.add_argument('--skeleton_file', type=str, required=True,
                       help='Cross-database SQL skeleton file')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='Database directory')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph',
                       help='Output directory for graph files')

    args = parser.parse_args()

    # Load cross-database SQL skeletons
    print(f"Loading cross-database SQL skeletons: {args.skeleton_file}")
    with open(args.skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)

    print(f"Total SQL skeletons: {len(skeletons)}")

    # Collect all involved databases
    all_databases = set()
    for skeleton in skeletons:
        all_databases.update(skeleton.get('databases', []))

    print(f"Involved databases: {sorted(all_databases)}")

    # Load schemas from all databases
    print("\nLoading database schemas...")
    schemas = load_multiple_schemas(all_databases, args.database_dir)
    print(f"Successfully loaded schemas from {len(schemas)} databases")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each SQL skeleton
    print(f"\nGenerating graph files...")
    success_count = 0
    for skeleton in tqdm(skeletons, desc="Generating graphs"):
        try:
            output_file = process_cross_database_skeleton(
                skeleton,
                schemas,
                args.output_dir
            )
            success_count += 1
        except Exception as e:
            print(f"\nFailed to process skeleton: {e}")
            continue

    print(f"\nDone! Successfully generated {success_count}/{len(skeletons)} graph files")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()
