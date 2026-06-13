#!/usr/bin/env python3
"""
Improved SQL-Schema Linking Graph construction script

Improvements:
1. Preserve original graph construction logic
2. Enhance graph information: add table descriptions, column data types, and other metadata
3. Better handling of foreign key relations
4. Prepare for subsequent graph utilization
"""

import json
import networkx as nx
import re
import os
from tqdm import tqdm
from collections import defaultdict

def load_schema(schema_file):
    """Load database schema information"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if standard schema format (contains 'tables' key)
    if 'tables' in data:
        return data
    
    # If not standard format, extract schema from database JSON file
    # Database JSON format: {table_name: {columns: [...], data: [...]}}
    schema = {'tables': []}
    for table_name, table_data in data.items():
        columns = []
        if 'columns' in table_data:
            # Extract column names from columns list
            for col_name in table_data['columns']:
                # Infer data type (default TEXT)
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'  # Default type
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

def parse_sql_framework(sql_framework):
    """
    Parse SQL framework, identifying placeholder positions and types.
    Returns a list where each element is {'clause': clause type, 'text': placeholder text, 'position': position info}.
    """
    placeholders = []
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'JOIN', 'ON', 'UNION']
    pattern = re.compile(r'\b(' + '|'.join(clauses) + r')\b', re.IGNORECASE)
    tokens = pattern.split(sql_framework)
    current_clause = 'UNKNOWN'
    position = 0
    
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.upper() in clauses:
            current_clause = token.upper()
        else:
            # Find placeholders in current clause
            placeholder_matches = re.finditer(r'(_+)', token)
            for match in placeholder_matches:
                placeholders.append({
                    'clause': current_clause,
                    'text': match.group(1),
                    'position': position,
                    'context': token[:match.start()] + token[match.end():]  # Context around placeholder
                })
                position += 1
    return placeholders

def get_possible_schema_nodes(G, placeholder_type):
    """
    Get possible schema nodes to connect based on placeholder type.
    Returns a list of cleaned node IDs.
    """
    nodes = []
    if placeholder_type == 'SELECT':
        # SELECT clause, may connect to column nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'column']
    elif placeholder_type == 'FROM':
        # FROM clause, may connect to table nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'table']
    elif placeholder_type in ['WHERE', 'HAVING']:
        # WHERE/HAVING clause, may connect to column nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'column']
    elif placeholder_type in ['GROUP BY', 'ORDER BY']:
        # GROUP BY/ORDER BY clause, may connect to column nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'column']
    elif placeholder_type == 'JOIN':
        # JOIN clause, may connect to table nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'table']
    elif placeholder_type == 'ON':
        # ON clause, may connect to column nodes (for JOIN conditions)
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'column']
    else:
        # Other clause types, default to table and column nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') in ('table', 'column')]
    return nodes

def clean_xml_string(s):
    """Clean string, remove XML-incompatible characters"""
    if not isinstance(s, str):
        s = str(s)
    # Remove NULL bytes and control characters (keep newlines and tabs)
    # XML disallowed control characters: 0x00-0x1F (except 0x09, 0x0A, 0x0D)
    s = ''.join(char for char in s if ord(char) >= 32 or char in '\n\t\r')
    # Remove NULL bytes
    s = s.replace('\x00', '')
    # Remove other control characters
    s = ''.join(char for char in s if ord(char) >= 32 or char in '\n\t\r')
    return s

def clean_node_id(node_id):
    """Clean node ID for XML compatibility"""
    if not isinstance(node_id, str):
        node_id = str(node_id)
    # Node ID cannot contain certain special characters; use stricter cleaning
    # Keep only alphanumeric, underscore, and dot
    cleaned = ''.join(char for char in node_id if char.isalnum() or char in '._-')
    # If empty after cleaning, use hash value
    if not cleaned:
        cleaned = f"node_{abs(hash(node_id)) % 1000000}"
    return cleaned

def build_sql_schema_graph(schema_info, sql_framework):
    """
    Build SQL schema graph (improved version).
    Contains relationships between tables, columns, and SQL placeholders with enhanced metadata.
    """
    G = nx.Graph()

    # 1. Add table and column nodes (enhanced metadata)
    # Create mapping from node ID to original name
    node_id_map = {}  # {cleaned_id: original_name}
    
    for table in schema_info['tables']:
        table_name = table['table_name']
        table_comment = clean_xml_string(table.get('table_comment', ''))
        table_description = clean_xml_string(table.get('table_description', 'No description available.'))
        
        # Clean table node ID
        table_node_id = clean_node_id(table_name)
        node_id_map[table_node_id] = table_name
        
        # Add table node with description info
        G.add_node(table_node_id, 
                  node_type='table', 
                  label=clean_xml_string(table_name),
                  original_name=clean_xml_string(table_name),
                  comment=table_comment,
                  description=table_description)

        # Add column nodes
        for column in table['columns']:
            column_name = column['column_name']
            data_type = column.get('data_type', 'TEXT')
            full_column_name = f"{table_name}.{column_name}"
            
            # Clean column node ID
            column_node_id = clean_node_id(full_column_name)
            node_id_map[column_node_id] = full_column_name
            
            # Clean all string attributes
            G.add_node(column_node_id, 
                      node_type='column', 
                      label=clean_xml_string(full_column_name),
                      original_name=clean_xml_string(full_column_name),
                      table=clean_xml_string(table_name),
                      column=clean_xml_string(column_name),
                      data_type=clean_xml_string(data_type))

            # Table-column edge
            G.add_edge(table_node_id, column_node_id, edge_type='table-column')

    # 2. Add foreign key edges (enhanced info)
    for fk in schema_info.get('foreign_keys', []):
        source_table = fk['table']
        source_column = fk['column']
        target_table = fk['references']['table']
        target_column = fk['references']['column']

        source_node_original = f"{source_table}.{source_column}"
        target_node_original = f"{target_table}.{target_column}"
        
        # Use cleaned node IDs
        source_node = clean_node_id(source_node_original)
        target_node = clean_node_id(target_node_original)

        if source_node in G.nodes and target_node in G.nodes:
            # Add foreign key edge with relation info
            G.add_edge(source_node, target_node, 
                      edge_type='foreign-key',
                      source_table=clean_xml_string(source_table),
                      target_table=clean_xml_string(target_table),
                      relationship='references')

    # 3. Parse SQL framework, add SQL placeholder nodes
    sql_placeholders = parse_sql_framework(sql_framework)
    for idx, placeholder in enumerate(sql_placeholders):
        placeholder_type = placeholder.get('clause', 'UNKNOWN')
        placeholder_text = placeholder.get('text', '_')
        sql_node = f"sql_placeholder_{idx}"  # This ID is already safe
        
        G.add_node(sql_node, 
                  node_type='sql_placeholder', 
                  placeholder_type=clean_xml_string(placeholder_type),
                  placeholder_text=clean_xml_string(placeholder_text),
                  position=placeholder.get('position', idx),
                  context=clean_xml_string(placeholder.get('context', '')))

        # Connect to possible schema nodes based on placeholder type
        possible_nodes = get_possible_schema_nodes(G, placeholder_type)
        for node in possible_nodes:
            G.add_edge(sql_node, node, edge_type='sql-schema')
    
    # Note: node_id_map cannot be saved directly to GraphML (dict type not supported)
    # If needed, save to a separate JSON file
    # Original names can be recovered via the original_name attribute on nodes

    return G, node_id_map

def extract_graph_metadata(G):
    """
    Extract metadata from graph for subsequent intelligent reasoning.
    Returns:
    - foreign_key_relations: list of foreign key relations
    - table_info: table info dict
    - column_info: column info dict
    """
    foreign_key_relations = []
    table_info = {}
    column_info = {}
    
    for node, attr in G.nodes(data=True):
        node_type = attr.get('node_type')
        if node_type == 'table':
            table_info[node] = {
                'name': node,
                'comment': attr.get('comment', ''),
                'description': attr.get('description', 'No description available.'),
                'columns': []
            }
        elif node_type == 'column':
            table_name = attr.get('table', '')
            column_name = attr.get('column', '')
            column_info[node] = {
                'full_name': node,
                'table': table_name,
                'column': column_name,
                'data_type': attr.get('data_type', 'TEXT')
            }
            if table_name in table_info:
                table_info[table_name]['columns'].append(node)
    
    # Extract foreign key relations
    for u, v, data in G.edges(data=True):
        if data.get('edge_type') == 'foreign-key':
            foreign_key_relations.append({
                'source': u,
                'target': v,
                'source_table': data.get('source_table', ''),
                'target_table': data.get('target_table', '')
            })
    
    return {
        'foreign_key_relations': foreign_key_relations,
        'table_info': table_info,
        'column_info': column_info
    }

def save_graph(G, output_file):
    """Save graph in GraphML format"""
    nx.write_graphml(G, output_file)
    # Do not print per-file save info to reduce output

def save_graph_metadata(metadata, output_file):
    """Save graph metadata as JSON"""
    # Convert sets to lists for JSON serialization
    serializable_metadata = {
        'foreign_key_relations': metadata['foreign_key_relations'],
        'table_info': {k: {
            'name': v['name'],
            'comment': v['comment'],
            'description': v['description'],
            'columns': v['columns']
        } for k, v in metadata['table_info'].items()},
        'column_info': metadata['column_info']
    }
    
    # Add node_id_map (if present)
    if 'node_id_map' in metadata:
        serializable_metadata['node_id_map'] = metadata['node_id_map']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_metadata, f, ensure_ascii=False, indent=2)
    # Do not print per-file save info to reduce output

def process_database(database_name, skeleton_file, schema_file, output_dir):
    """
    Process a single database and generate graph files.
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
    
    # Build graph for each SQL skeleton
    for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} progress", leave=False)):
        # If string, use directly; if dict, extract sql_framework field
        if isinstance(sql_skeleton, dict):
            sql_framework = sql_skeleton.get('sql_framework', '')
        else:
            sql_framework = sql_skeleton
        
        if not sql_framework:
            continue
        
        # Build graph
        G, node_id_map = build_sql_schema_graph(schema, sql_framework)
        
        # Extract metadata
        metadata = extract_graph_metadata(G)
        
        # Add node_id_map to metadata
        metadata['node_id_map'] = node_id_map
        
        # Save graph
        graph_file = os.path.join(graph_dir, f"{database_name}_graph_{idx}.graphml")
        save_graph(G, graph_file)
        
        # Save metadata (includes node_id_map)
        metadata_file = os.path.join(graph_dir, f"{database_name}_metadata_{idx}.json")
        save_graph_metadata(metadata, metadata_file)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build SQL-Schema Linking Graph (improved version)')
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
    print("✓ Graph construction complete for all databases!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
