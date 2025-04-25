import json
import networkx as nx
import re
import os
from tqdm import tqdm

def load_schema(schema_file):
    """加载数据库的 schema 信息"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

def load_sql_frameworks(sql_frameworks_file, is_old_database=False):
    """
    加载 SQL 框架。
    如果是旧数据库，则提取每个条目的 'sql_framework' 字段。
    如果是新数据库，假设文件是一个字典，键为数据库名，值为框架列表；或者是一个列表，属于默认数据库。
    """
    with open(sql_frameworks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if is_old_database:
        # 旧数据库的文件是一个列表，每个元素是一个包含多个字段的字典
        # 提取每个元素的 'sql_framework' 字段
        sql_frameworks = [item['sql_framework'] for item in data if 'sql_framework' in item]
    else:
        # 新数据库的文件现在是每个数据库一个文件，不需要在这里处理
        # 处理方式将在 process_new_database 函数中
        sql_frameworks = None
    return sql_frameworks

def build_sql_schema_graph(schema_info, sql_framework):
    """
    构建 SQL schema 图。
    包含表、列和 SQL 占位符之间的关系。
    """
    G = nx.Graph()

    # 添加模式节点和边
    for table in schema_info['tables']:
        table_name = table['table_name']
        G.add_node(table_name, node_type='table', label=table_name)

        for column in table['columns']:
            column_name = column['column_name']
            full_column_name = f"{table_name}.{column_name}"
            G.add_node(full_column_name, node_type='column', label=full_column_name)

            # 表-列边
            G.add_edge(table_name, full_column_name, edge_type='table-column')

    # 添加外键边
    for fk in schema_info.get('foreign_keys', []):
        source_table = fk['table']
        source_column = fk['column']
        target_table = fk['references']['table']
        target_column = fk['references']['column']

        source_node = f"{source_table}.{source_column}" 
        target_node = f"{target_table}.{target_column}"

        if source_node in G.nodes and target_node in G.nodes:
            G.add_edge(source_node, target_node, edge_type='foreign-key')

    # 解析 SQL 框架，添加 SQL 占位符节点
    sql_placeholders = parse_sql_framework(sql_framework)
    for idx, placeholder in enumerate(sql_placeholders):
        placeholder_type = placeholder.get('clause', 'UNKNOWN')
        placeholder_text = placeholder.get('text', '_')
        sql_node = f"sql_{idx}"
        G.add_node(sql_node, node_type='sql_placeholder', placeholder_type=placeholder_type, label=placeholder_text)

        # 根据占位符类型，连接到可能的模式节点
        possible_nodes = get_possible_schema_nodes(G, placeholder_type)
        for node in possible_nodes:
            G.add_edge(sql_node, node, edge_type='sql-schema')

    return G

def parse_sql_framework(sql_framework):
    """
    解析 SQL 框架，识别占位符的位置和类型。
    返回列表，每个元素为 {'clause': 子句类型, 'text': 占位符文本}。
    """
    placeholders = []
    # 使用正则表达式解析 SQL 框架，识别子句和占位符
    clauses = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT', 'JOIN', 'ON']
    pattern = re.compile(r'\b(' + '|'.join(clauses) + r')\b', re.IGNORECASE)
    tokens = pattern.split(sql_framework)
    current_clause = 'UNKNOWN'  # 初始化为默认值
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token.upper() in clauses:
            current_clause = token.upper()
        else:
            # 在当前子句中查找占位符
            placeholder_matches = re.findall(r'(_+)', token)
            for placeholder in placeholder_matches:
                placeholders.append({'clause': current_clause, 'text': placeholder})
    return placeholders

def get_possible_schema_nodes(G, placeholder_type):
    """
    根据占位符类型，获取可能连接的模式节点。
    """
    nodes = []
    if placeholder_type == 'SELECT':
        # SELECT 子句，可能连接到列节点
        nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'column']
    elif placeholder_type == 'FROM':
        # FROM 子句，可能连接到表节点
        nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'table']
    elif placeholder_type == 'WHERE':
        # WHERE 子句，可能连接到列节点和值节点（这里暂时没有值节点）
        nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] == 'column']
    else:
        # 其他子句类型，根据需要添加
        nodes = [n for n, attr in G.nodes(data=True) if attr['node_type'] in ('table', 'column')]
    return nodes

def save_graph(G, output_file):
    """保存图为 GraphML 格式"""
    nx.write_graphml(G, output_file)
    print(f"连结图已保存到 {output_file}")

def process_old_database():
    """处理旧数据库，生成连结图"""
    old_schema_file = '../../data/12345_database_schema.json'  # 更新路径
    old_sql_frameworks_file = '../../data/old_ast_cfg.json'  # 保持不变
    old_schema = load_schema(old_schema_file)
    old_sql_frameworks = load_sql_frameworks(old_sql_frameworks_file, is_old_database=True)

    old_graph_dir = '../../data/old_graph'
    os.makedirs(old_graph_dir, exist_ok=True)

    print("正在处理旧数据库的 SQL 框架...")
    # 使用 tqdm 添加进度条
    for idx, sql_framework in enumerate(tqdm(old_sql_frameworks, desc="旧数据库框架处理进度")):
        output_file = os.path.join(old_graph_dir, f"old_graph_{idx}.graphml")
        if os.path.exists(output_file):
            print(f"{output_file} 已存在，跳过。")
            continue
        G = build_sql_schema_graph(old_schema, sql_framework)
        save_graph(G, output_file)

def process_new_database():
    """处理新数据库，生成连结图"""
    new_skeleton_dir = '../../data/new_sql_skeleton/'
    new_structure_dir = '../../data/new_sql_structure/'
    new_graph_base_dir = '../../data/new_graph/'
    new_schema_dir = '../../data/new_schema/'

    # 确保输出基目录存在
    os.makedirs(new_graph_base_dir, exist_ok=True)

    # 遍历 new_sql_skeleton 目录下的所有数据库 skeleton 文件
    skeleton_files = [f for f in os.listdir(new_skeleton_dir) if f.endswith('_sql_skeleton.json')]

    print("正在处理新数据库的 SQL skeleton 文件...")

    for skeleton_file in tqdm(skeleton_files, desc="新数据库处理进度"):
        # 提取数据库名称
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        skeleton_file_path = os.path.join(new_skeleton_dir, skeleton_file)

        # 加载对应的 SQL skeletons
        with open(skeleton_file_path, 'r', encoding='utf-8') as f:
            sql_skeletons = json.load(f)

        # 加载对应的 schema 文件
        schema_file = os.path.join(new_schema_dir, f"{database_name}_schema.json")
        if not os.path.exists(schema_file):
            print(f"Schema file for database '{database_name}' does not exist. Skipping.")
            continue
        new_schema = load_schema(schema_file)

        # 创建数据库对应的 graph 文件夹
        graph_dir = os.path.join(new_graph_base_dir, database_name)
        os.makedirs(graph_dir, exist_ok=True)

        # 使用 tqdm 添加每个数据库内部的 SQL 框架处理进度
        print(f"正在处理数据库 '{database_name}' 的 SQL skeleton...")
        for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} 的框架处理进度", leave=False)):
            output_file = os.path.join(graph_dir, f"{database_name}_graph_{idx}.graphml")
            if os.path.exists(output_file):
                print(f"{output_file} 已存在，跳过。")
                continue

            # 构建 schema 图
            G = build_sql_schema_graph(new_schema, sql_skeleton)
            save_graph(G, output_file)

def main():
    """主函数，处理旧数据库和新数据库"""
    # 处理旧数据库
    process_old_database()
    # 处理新数据库
    process_new_database()

if __name__ == '__main__':
    main()
