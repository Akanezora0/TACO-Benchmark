#!/usr/bin/env python3
"""
Extract key information from graph files to guide SQL skeleton filling.

Core approach:
1. Analyze SQL skeleton to determine required node types
2. Extract nodes directly related to SQL placeholders from the graph
3. Extract foreign key relations between these nodes
4. Include only relevant tables and columns to significantly reduce prompt size
"""

import networkx as nx
import json
import re
from collections import defaultdict

def analyze_sql_framework(sql_framework):
    """
    Analyze SQL skeleton to determine required table and column types.
    Returns: required node types and counts.
    """
    sql_upper = sql_framework.upper()
    
    # Analyze required table count
    has_join = 'JOIN' in sql_upper
    has_union = 'UNION' in sql_upper
    has_subquery = '(' in sql_framework and 'SELECT' in sql_upper
    
    # Count placeholders
    placeholder_count = sql_framework.count('_')
    
    # Estimate required table count
    if has_join:
        # JOIN typically requires 2-3 tables
        num_joins = sql_upper.count('JOIN')
        estimated_tables = min(num_joins + 1, 5)
    elif has_union:
        # UNION typically requires 2 tables
        estimated_tables = 2
    elif has_subquery:
        # Subqueries may need 2-3 tables
        estimated_tables = min(3, placeholder_count // 3)
    else:
        # Single-table query
        estimated_tables = 1
    
    # Analyze required column types
    needs_select_columns = 'SELECT' in sql_upper
    needs_where_columns = 'WHERE' in sql_upper
    needs_groupby_columns = 'GROUP BY' in sql_upper
    needs_orderby_columns = 'ORDER BY' in sql_upper
    
    return {
        'estimated_tables': estimated_tables,
        'has_join': has_join,
        'has_union': has_union,
        'has_subquery': has_subquery,
        'needs_select_columns': needs_select_columns,
        'needs_where_columns': needs_where_columns,
        'needs_groupby_columns': needs_groupby_columns,
        'needs_orderby_columns': needs_orderby_columns,
        'placeholder_count': placeholder_count
    }

def extract_relevant_nodes_from_graph(G, sql_framework, max_tables=5, max_columns_per_table=10):
    """
    Extract key nodes from graph file related to the SQL skeleton.
    
    Strategy:
    1. Find all SQL placeholder nodes
    2. Find table and column nodes directly connected to placeholders
    3. Select most relevant tables based on SQL skeleton semantics
    4. Extract columns and foreign key relations for these tables
    """
    if G is None:
        return None
    
    # Analyze SQL skeleton
    sql_analysis = analyze_sql_framework(sql_framework)
    
    # 1. Find all SQL placeholder nodes
    sql_placeholders = []
    for node, attr in G.nodes(data=True):
        if attr.get('node_type') == 'sql_placeholder':
            sql_placeholders.append({
                'node_id': node,
                'placeholder_type': attr.get('placeholder_type', 'UNKNOWN'),
                'position': attr.get('position', 0)
            })
    
    if not sql_placeholders:
        return None
    
    # 2. Find all nodes directly connected to placeholders
    connected_nodes = set()
    placeholder_connections = defaultdict(list)
    
    for placeholder in sql_placeholders:
        placeholder_id = placeholder['node_id']
        neighbors = list(G.neighbors(placeholder_id))
        connected_nodes.update(neighbors)
        placeholder_connections[placeholder_id] = neighbors
    
    # 3. Extract tables and columns from connected nodes
    relevant_tables = set()
    relevant_columns = set()
    table_to_columns = defaultdict(set)
    
    for node_id in connected_nodes:
        node_attr = G.nodes[node_id]
        node_type = node_attr.get('node_type')
        
        if node_type == 'table':
            table_name = node_attr.get('original_name') or node_attr.get('label') or node_id
            relevant_tables.add(table_name)
        elif node_type == 'column':
            column_name = node_attr.get('original_name') or node_attr.get('label') or node_id
            table_name = node_attr.get('table', '')
            if table_name:
                relevant_tables.add(table_name)
                relevant_columns.add(column_name)
                table_to_columns[table_name].add(column_name)
    
    # 4. If too many tables, prioritize those with foreign key relations
    if len(relevant_tables) > max_tables:
        # Extract foreign key relations, prioritize tables with FK relations
        fk_relations = []
        for u, v, attr in G.edges(data=True):
            if attr.get('edge_type') == 'foreign-key':
                source_table = attr.get('source_table', '')
                target_table = attr.get('target_table', '')
                if source_table in relevant_tables and target_table in relevant_tables:
                    fk_relations.append((source_table, target_table))
        
        # Build table relation graph
        table_graph = defaultdict(set)
        for source, target in fk_relations:
            table_graph[source].add(target)
            table_graph[target].add(source)
        
        # Prioritize tables with foreign key relations
        prioritized_tables = set()
        for source, target in fk_relations[:max_tables]:
            prioritized_tables.add(source)
            prioritized_tables.add(target)
            if len(prioritized_tables) >= max_tables:
                break
        
        # If still not enough, randomly add other tables
        if len(prioritized_tables) < max_tables:
            remaining = list(relevant_tables - prioritized_tables)
            prioritized_tables.update(remaining[:max_tables - len(prioritized_tables)])
        
        relevant_tables = prioritized_tables
    
    # 5. Limit column count per table
    filtered_table_to_columns = {}
    for table in relevant_tables:
        columns = list(table_to_columns.get(table, []))[:max_columns_per_table]
        if columns:
            filtered_table_to_columns[table] = columns
    
    # 6. Extract foreign key relations (only for relevant tables)
    foreign_keys = []
    for u, v, attr in G.edges(data=True):
        if attr.get('edge_type') == 'foreign-key':
            source_table = attr.get('source_table', '')
            target_table = attr.get('target_table', '')
            if source_table in relevant_tables and target_table in relevant_tables:
                # Get column info from node attributes
                source_node_attr = G.nodes.get(u, {})
                target_node_attr = G.nodes.get(v, {})
                source_column = source_node_attr.get('column', '') or source_node_attr.get('original_name', '').split('.')[-1]
                target_column = target_node_attr.get('column', '') or target_node_attr.get('original_name', '').split('.')[-1]
                
                foreign_keys.append({
                    'source_table': source_table,
                    'source_column': source_column,
                    'target_table': target_table,
                    'target_column': target_column
                })
    
    # 7. Extract table information (description, comment, etc.)
    table_info = {}
    for node_id in G.nodes():
        node_attr = G.nodes[node_id]
        if node_attr.get('node_type') == 'table':
            table_name = node_attr.get('original_name') or node_attr.get('label') or node_id
            if table_name in relevant_tables:
                table_info[table_name] = {
                    'name': table_name,
                    'comment': node_attr.get('comment', ''),
                    'description': node_attr.get('description', 'No description available.')
                }
    
    # 8. Extract column information (data types, etc.)
    column_info = {}
    for node_id in G.nodes():
        node_attr = G.nodes[node_id]
        if node_attr.get('node_type') == 'column':
            column_name = node_attr.get('original_name') or node_attr.get('label') or node_id
            table_name = node_attr.get('table', '')
            if table_name in relevant_tables and column_name in filtered_table_to_columns.get(table_name, []):
                column_info[column_name] = {
                    'full_name': column_name,
                    'table': table_name,
                    'column': node_attr.get('column', ''),
                    'data_type': node_attr.get('data_type', 'TEXT')
                }
    
    return {
        'tables': list(relevant_tables),
        'table_info': table_info,
        'columns': filtered_table_to_columns,
        'column_info': column_info,
        'foreign_keys': foreign_keys,
        'sql_analysis': sql_analysis
    }

def format_extracted_info_for_prompt(extracted_info, sql_framework):
    """
    Format extracted information as text suitable for prompts.
    """
    if not extracted_info:
        return ""
    
    tables = extracted_info.get('tables', [])
    table_info = extracted_info.get('table_info', {})
    columns = extracted_info.get('columns', {})
    column_info = extracted_info.get('column_info', {})
    foreign_keys = extracted_info.get('foreign_keys', [])
    
    prompt_text = "\n=== Database Schema Information (Compact) ===\n\n"
    
    # Table information
    prompt_text += f"Available tables ({len(tables)} total):\n"
    for table in tables:
        prompt_text += f"- {table}\n"
    
    # Table details
    if table_info:
        prompt_text += "\nTable Details:\n"
        for table_name in tables:
            if table_name in table_info:
                info = table_info[table_name]
                prompt_text += f"\nTable: {table_name}\n"
                if info.get('description') and info['description'] != 'No description available.':
                    prompt_text += f"Description: {info['description']}\n"
                if info.get('comment'):
                    prompt_text += f"Comment: {info['comment']}\n"
                
                # Column information
                if table_name in columns:
                    prompt_text += "Columns:\n"
                    for col_name in columns[table_name]:
                        if col_name in column_info:
                            col_info = column_info[col_name]
                            data_type = col_info.get('data_type', 'TEXT')
                            prompt_text += f"  - {col_name} (Type: {data_type})\n"
    
    # Foreign key relations
    if foreign_keys:
        prompt_text += "\nForeign Key Relations (usable for JOIN):\n"
        for fk in foreign_keys:
            prompt_text += f"- {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}\n"
    
    return prompt_text

def extract_and_save_relevant_info(graph_file, sql_framework, output_file=None):
    """
    Extract key information from graph file and save.
    """
    # Load graph file
    try:
        G = nx.read_graphml(graph_file)
    except Exception as e:
        print(f"Failed to load graph file: {e}")
        return None
    
    # Extract key information
    extracted_info = extract_relevant_nodes_from_graph(G, sql_framework)
    
    if not extracted_info:
        return None
    
    # Add SQL skeleton information
    extracted_info['sql_framework'] = sql_framework
    
    # Save to file (if output file specified)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_info, f, ensure_ascii=False, indent=2)
    
    return extracted_info

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 graph_extractor.py <graph_file> <sql_framework> [output_file]")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    sql_framework = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    extracted_info = extract_and_save_relevant_info(graph_file, sql_framework, output_file)
    
    if extracted_info:
        print("Extraction successful!")
        print(f"Relevant table count: {len(extracted_info['tables'])}")
        print(f"Foreign key relation count: {len(extracted_info['foreign_keys'])}")
        
        # Formatted output
        formatted = format_extracted_info_for_prompt(extracted_info, sql_framework)
        print("\nFormatted information:")
        print(formatted)
    else:
        print("Extraction failed")
