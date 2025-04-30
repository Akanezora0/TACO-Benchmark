import json
import networkx as nx
import re
import os
from tqdm import tqdm

def load_schema(schema_file):
    """load the schema information of the database"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

def load_sql_frameworks(sql_frameworks_file, is_old_database=False):
    """
    load the SQL framework.
    if the database is old, extract the 'sql_framework' field of each item.
    if the database is new, assume the file is a dictionary, with the key as the database name and the value as the framework list; or a list, belongs to the default database.
    """
    with open(sql_frameworks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if is_old_database:
        # the file of old database is a list, each element is a dictionary containing multiple fields
        # extract the 'sql_framework' field of each element
        sql_frameworks = [item['sql_framework'] for item in data if 'sql_framework' in item]
    else:
        # the file of new database is now a file for each database, no need to handle here
        # the processing will be in the process_new_database function
        sql_frameworks = None
    return sql_frameworks

def build_sql_schema_graph(schema_info, sql_framework):
    """
    build the SQL schema graph.
    include the relationship between tables, columns and SQL placeholders.
    """
    G = nx.Graph()

    # add the schema nodes and edges
    for table in schema_info['tables']:
        table_name = table['table_name']
        G.add_node(table_name, node_type='table', label=table_name)

        for column in table['columns']:
            column_name = column['column_name']
            full_column_name = f"{table_name}.{column_name}"
            G.add_node(full_column_name, node_type='column', label=full_column_name)

            # table-column edge
            G.add_edge(table_name, full_column_name, edge_type='table-column')

    # add the foreign key edges
    for fk in schema_info.get('foreign_keys', []):
        source_table = fk['table']
        source_column = fk['column']
        target_table = fk['references']['table']
        target_column = fk['references']['column']

        source_node = f"{source_table}.{source_column}" 
        target_node = f"{target_table}.{target_column}"

        if source_node in G.nodes and target_node in G.nodes:
            G.add_edge(source_node, target_node, edge_type='foreign-key')

    # parse the SQL framework, add the SQL placeholder nodes
    sql_placeholders = parse_sql_framework(sql_framework)
    for idx, placeholder in enumerate(sql_placeholders):
        placeholder_type = placeholder.get('clause', 'UNKNOWN')
        placeholder_text = placeholder.get('text', '_')
        sql_node = f"sql_{idx}"
        G.add_node(sql_node, node_type='sql_placeholder', placeholder_type=placeholder_type, label=placeholder_text)

        # add the edges between the SQL placeholder and the possible schema nodes
        possible_nodes = get_possible_schema_nodes(G, placeholder_type)
        for node in possible_nodes:
            G.add_edge(sql_node, node, edge_type='sql-schema')

    return G

def parse_sql_framework(sql_framework):
    """
    parse the SQL framework, identify the position and type of the placeholder.
    return a list, each element is {'clause': clause type, 'text': placeholder text}.
    """
    placeholders = []
    # use the regular expression to parse the SQL framework, identify the clause and placeholder
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'JOIN', 'ON']
    pattern = re.compile(r'\b(' + '|'.join(clauses) + r')\b', re.IGNORECASE)
    tokens = pattern.split(sql_framework)
    current_clause = 'UNKNOWN'
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.upper() in clauses:
            current_clause = token.upper()
        else:
            placeholder_matches = re.findall(r'(_+)', token)
            for placeholder in placeholder_matches:
                placeholders.append({'clause': current_clause, 'text': placeholder})
    return placeholders

def get_possible_schema_nodes(G, placeholder_type):
    """
    get the possible schema nodes according to the placeholder type.
    """
    nodes = []
    if placeholder_type == 'SELECT':
        # SELECT clause, possible connected to column nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'column']
    elif placeholder_type == 'FROM':
        # FROM clause, possible connected to table nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'table']
    elif placeholder_type == 'WHERE':
        # WHERE clause, possible connected to column nodes and value nodes
        nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'column']
    else:
        # other clause types, add according to the need
        nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] in ('table', 'column')]
    return nodes

def save_graph(G, output_file):
    """save the graph as the GraphML format"""
    nx.write_graphml(G, output_file)
    print(f"The graph has been saved to {output_file}")

def process_old_database():
    """process the old database, generate the connected graph"""
    old_schema_file = '../../data/12345_database_schema.json'  # update the path
    old_sql_frameworks_file = '../../data/old_ast_cfg.json'  # keep the same
    old_schema = load_schema(old_schema_file)
    old_sql_frameworks = load_sql_frameworks(old_sql_frameworks_file, is_old_database=True)

    old_graph_dir = '../../data/old_graph'
    os.makedirs(old_graph_dir, exist_ok=True)

    print("Processing the old database SQL framework...")
    # use tqdm to add the progress bar
    for idx, sql_framework in enumerate(tqdm(old_sql_frameworks, desc="Old database framework processing progress")):
        output_file = os.path.join(old_graph_dir, f"old_graph_{idx}.graphml")
        if os.path.exists(output_file):
            print(f"{output_file} already exists, skip.")
            continue
        G = build_sql_schema_graph(old_schema, sql_framework)
        save_graph(G, output_file)

def process_new_database():
    """process the new database, generate the connected graph"""
    new_skeleton_dir = '../../data/new_sql_skeleton/'
    new_structure_dir = '../../data/new_sql_structure/'
    new_graph_base_dir = '../../data/new_graph/'
    new_schema_dir = '../../data/new_schema/'

    # ensure the output base directory exists
    os.makedirs(new_graph_base_dir, exist_ok=True)

    # traverse all the database skeleton files in the new_sql_skeleton directory
    skeleton_files = [f for f in os.listdir(new_skeleton_dir) if f.endswith('_sql_skeleton.json')]

    print("Processing the new database SQL skeleton files...")

    for skeleton_file in tqdm(skeleton_files, desc="New database processing progress"):
        # extract the database name
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        skeleton_file_path = os.path.join(new_skeleton_dir, skeleton_file)

        # load the corresponding SQL skeletons
        with open(skeleton_file_path, 'r', encoding='utf-8') as f:
            sql_skeletons = json.load(f)

        # load the corresponding schema file
        schema_file = os.path.join(new_schema_dir, f"{database_name}_schema.json")
        if not os.path.exists(schema_file):
            print(f"Schema file for database '{database_name}' does not exist. Skipping.")
            continue
        new_schema = load_schema(schema_file)

        # create the graph file folder for the corresponding database
        graph_dir = os.path.join(new_graph_base_dir, database_name)
        os.makedirs(graph_dir, exist_ok=True)

        # use tqdm to add the progress bar for the SQL framework processing in each database
        print(f"Processing the SQL skeleton of database '{database_name}'...")
        for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} framework processing progress", leave=False)):
            output_file = os.path.join(graph_dir, f"{database_name}_graph_{idx}.graphml")
            if os.path.exists(output_file):
                print(f"{output_file} already exists, skip.")
                continue

            # build the schema graph
            G = build_sql_schema_graph(new_schema, sql_skeleton)
            save_graph(G, output_file)

def main():
    """main function, process the old database and the new database"""
    
    process_old_database()
    process_new_database()

if __name__ == '__main__':
    main()
