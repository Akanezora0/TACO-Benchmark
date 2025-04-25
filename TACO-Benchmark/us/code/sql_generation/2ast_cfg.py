import os
import json
import sqlglot
from sqlglot import parse_one
import pandas as pd

def get_cfg_rules(node, rules=None):
    """
    递归遍历AST，生成CFG规则序列
    """
    if rules is None:
        rules = []

    if not node:
        return rules

    rule = node.key
    children = []
    # 处理 node 的所有子节点
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
    将AST节点转换为可序列化的字典形式
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
        print(f"处理文件 {input_file}，共有 {len(data_list)} 条记录。")

    processed_data = []
    for idx, data in enumerate(data_list):
        sql_text = data.get('sql_framework', '')
        if not sql_text:
            continue
        try:
            # 解析SQL，生成AST
            ast = parse_one(sql_text)
            # 生成CFG规则序列
            cfg_rules = get_cfg_rules(ast)
            # 将AST转换为字典形式
            ast_dict = ast_to_dict(ast)
            # 更新数据项
            data['ast'] = ast_dict
            data['cfg_rules'] = cfg_rules
            processed_data.append(data)
        except Exception as e:
            print(f"第 {idx} 条记录解析SQL时出错：{e}")
            continue

    # 将结果写入输出文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(processed_data, outfile, ensure_ascii=False, indent=2)
    print(f"AST和CFG规则序列生成完成，结果保存在 {output_file}")

def generate_cfg_for_databases(databases_dir, input_skeleton_file, output_dir):
    # 遍历 parsed_data/ 下的所有数据库文件夹
    for db_folder in os.listdir(databases_dir):
        db_folder_path = os.path.join(databases_dir, db_folder)
        if os.path.isdir(db_folder_path):  # 如果是文件夹，每个文件夹代表一个数据库
            # 对应的 SQL skeleton 文件统一读取
            input_file = input_skeleton_file  # 统一的 SQL skeletons 文件
            output_file = os.path.join(output_dir, f"{db_folder}_ast_cfg.json")  # 输出文件路径

            # 生成 CFG 文件
            process_sql_file(input_file, output_file)
            print(f"{db_folder} 的 CFG 文件已生成并保存到 {output_file}")

if __name__ == '__main__':
    # 定义路径
    parsed_data_dir = os.path.join('..', '..', 'data', 'parsed_data')  # 使用已处理数据的 parsed_data 目录
    input_skeleton_file = os.path.join('..', '..', 'data', 'new_sql_skeletons.json')  # 统一的 SQL skeleton 文件
    output_dir = os.path.join('..', '..', 'data', 'new_asf_cfg')  # 生成的 CFG 文件保存目录

    # 创建保存 CFG 文件的目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 处理所有数据库
    generate_cfg_for_databases(parsed_data_dir, input_skeleton_file, output_dir)
