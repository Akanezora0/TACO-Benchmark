import os
import json
import sqlglot
from sqlglot import parse_one
import pandas as pd

def get_cfg_rules(node, rules=None):
    """
    recursively traverse the AST, generate the CFG rules sequence
    """
    if rules is None:
        rules = []

    if not node:
        return rules

    rule = node.key
    children = []
    # process all child nodes of node
    for arg_value in node.args.values():
        if isinstance(arg_value, sqlglot.Expression):
            children.append(arg_value.key)
        elif isinstance(arg_value, list):
            for item in arg_value:
                if isinstance(item, sqlglot.Expression):
                    children.append(item.key)
    if children:
        rule_str = f"{rule} -> {' '.join(children)}"
        rules.append(rule_str)
        for arg_value in node.args.values():
            if isinstance(arg_value, sqlglot.Expression):
                get_cfg_rules(arg_value, rules)
            elif isinstance(arg_value, list):
                for item in arg_value:
                    if isinstance(item, sqlglot.Expression):
                        get_cfg_rules(item, rules)
    else:
        rule_str = f"{rule} -> terminal"
        rules.append(rule_str)

    return rules

def ast_to_dict(node):
    """
    convert the AST node to a serializable dictionary
    """
    if not node:
        return None
    node_dict = {
        'type': node.key,
        'args': {}
    }
    for key, value in node.args.items():
        if isinstance(value, sqlglot.Expression):
            node_dict['args'][key] = ast_to_dict(value)
        elif isinstance(value, list):
            node_dict['args'][key] = [ast_to_dict(item) if isinstance(item, sqlglot.Expression) else str(item) for item in value]
        else:
            node_dict['args'][key] = str(value)
    return node_dict

def process_sql_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data_list = json.load(infile)
        print(f"Processing file {input_file}, containing {len(data_list)} records.")

    processed_data = []
    for idx, data in enumerate(data_list):
        sql_text = data.get('sql_framework', '')
        if not sql_text:
            continue
        try:
            ast = parse_one(sql_text)
            cfg_rules = get_cfg_rules(ast)
            ast_dict = ast_to_dict(ast)
            data['ast'] = ast_dict
            data['cfg_rules'] = cfg_rules
            processed_data.append(data)
        except Exception as e:
            print(f"Error parsing SQL at record {idx}: {e}")
            continue

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)
    print(f"AST and CFG rules sequence generation completed, results saved in {output_file}")

def generate_cfg_for_databases(databases_dir, input_skeleton_file, output_dir):
    for db_folder in os.listdir(databases_dir):
        db_folder_path = os.path.join(databases_dir, db_folder)
        if os.path.isdir(db_folder_path):  # if it is a folder, each folder represents a database
            input_file = input_skeleton_file
            output_file = os.path.join(output_dir, f"{db_folder}_ast_cfg.json")


            process_sql_file(input_file, output_file)
            print(f"{db_folder} CFG file generated and saved in {output_file}")

if __name__ == '__main__':
    # define the paths
    parsed_data_dir = os.path.join('..', '..', 'data', 'parsed_data')
    input_skeleton_file = os.path.join('..', '..', 'data', 'new_sql_skeletons.json')
    output_dir = os.path.join('..', '..', 'data', 'new_ast_cfg')  # the directory to save the generated CFG files

    os.makedirs(output_dir, exist_ok=True)

    # process all databases
    generate_cfg_for_databases(parsed_data_dir, input_skeleton_file, output_dir)
