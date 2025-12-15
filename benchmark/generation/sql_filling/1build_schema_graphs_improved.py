#!/usr/bin/env python3
"""
改进的SQL-Schema Linking Graph构建脚本

改进点：
1. 保留原有的图构建逻辑
2. 增强图的信息：添加表描述、列数据类型等元数据
3. 更好地处理外键关系
4. 为后续的图利用做准备
"""

import json
import networkx as nx
import re
import os
from tqdm import tqdm
from collections import defaultdict

def load_schema(schema_file):
    """加载数据库的schema信息"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否是标准schema格式（包含'tables'键）
    if 'tables' in data:
        return data
    
    # 如果不是标准格式，从数据库JSON文件中提取schema
    # 数据库JSON格式：{表名: {columns: [...], data: [...]}}
    schema = {'tables': []}
    for table_name, table_data in data.items():
        columns = []
        if 'columns' in table_data:
            # 从columns列表提取列名
            for col_name in table_data['columns']:
                # 尝试推断数据类型（默认为TEXT）
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'  # 默认类型
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
    解析SQL框架，识别占位符的位置和类型。
    返回列表，每个元素为 {'clause': 子句类型, 'text': 占位符文本, 'position': 位置信息}。
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
            # 在当前子句中查找占位符
            placeholder_matches = re.finditer(r'(_+)', token)
            for match in placeholder_matches:
                placeholders.append({
                    'clause': current_clause,
                    'text': match.group(1),
                    'position': position,
                    'context': token[:match.start()] + token[match.end():]  # 占位符周围的上下文
                })
                position += 1
    return placeholders

def get_possible_schema_nodes(G, placeholder_type):
    """
    根据占位符类型，获取可能连接的模式节点。
    返回清理后的节点ID列表。
    """
    nodes = []
    if placeholder_type == 'SELECT':
        # SELECT子句，可能连接到列节点
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'column']
    elif placeholder_type == 'FROM':
        # FROM子句，可能连接到表节点
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'table']
    elif placeholder_type in ['WHERE', 'HAVING']:
        # WHERE/HAVING子句，可能连接到列节点
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'column']
    elif placeholder_type in ['GROUP BY', 'ORDER BY']:
        # GROUP BY/ORDER BY子句，可能连接到列节点
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'column']
    elif placeholder_type == 'JOIN':
        # JOIN子句，可能连接到表节点
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'table']
    elif placeholder_type == 'ON':
        # ON子句，可能连接到列节点（用于JOIN条件）
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') == 'column']
    else:
        # 其他子句类型，默认连接到表和列节点
        nodes = [n for n, attr in G.nodes(data=True) if attr.get('node_type') in ('table', 'column')]
    return nodes

def clean_xml_string(s):
    """清理字符串，移除XML不兼容的字符"""
    if not isinstance(s, str):
        s = str(s)
    # 移除NULL字节和控制字符（保留换行符和制表符）
    # XML不允许的控制字符：0x00-0x1F（除了0x09, 0x0A, 0x0D）
    s = ''.join(char for char in s if ord(char) >= 32 or char in '\n\t\r')
    # 移除NULL字节
    s = s.replace('\x00', '')
    # 移除其他控制字符
    s = ''.join(char for char in s if ord(char) >= 32 or char in '\n\t\r')
    return s

def clean_node_id(node_id):
    """清理节点ID，确保XML兼容"""
    if not isinstance(node_id, str):
        node_id = str(node_id)
    # 节点ID不能包含某些特殊字符，使用更严格的清理
    # 移除所有控制字符和特殊字符，只保留字母、数字、下划线、点号
    cleaned = ''.join(char for char in node_id if char.isalnum() or char in '._-')
    # 如果清理后为空，使用哈希值
    if not cleaned:
        cleaned = f"node_{abs(hash(node_id)) % 1000000}"
    return cleaned

def build_sql_schema_graph(schema_info, sql_framework):
    """
    构建SQL schema图（改进版）。
    包含表、列和SQL占位符之间的关系，并增强元数据信息。
    """
    G = nx.Graph()

    # 1. 添加表节点和列节点（增强元数据）
    # 创建节点ID到原始名称的映射
    node_id_map = {}  # {cleaned_id: original_name}
    
    for table in schema_info['tables']:
        table_name = table['table_name']
        table_comment = clean_xml_string(table.get('table_comment', ''))
        table_description = clean_xml_string(table.get('table_description', 'No description available.'))
        
        # 清理表节点ID
        table_node_id = clean_node_id(table_name)
        node_id_map[table_node_id] = table_name
        
        # 添加表节点，包含描述信息
        G.add_node(table_node_id, 
                  node_type='table', 
                  label=clean_xml_string(table_name),
                  original_name=clean_xml_string(table_name),
                  comment=table_comment,
                  description=table_description)

        # 添加列节点
        for column in table['columns']:
            column_name = column['column_name']
            data_type = column.get('data_type', 'TEXT')
            full_column_name = f"{table_name}.{column_name}"
            
            # 清理列节点ID
            column_node_id = clean_node_id(full_column_name)
            node_id_map[column_node_id] = full_column_name
            
            # 清理所有字符串属性
            G.add_node(column_node_id, 
                      node_type='column', 
                      label=clean_xml_string(full_column_name),
                      original_name=clean_xml_string(full_column_name),
                      table=clean_xml_string(table_name),
                      column=clean_xml_string(column_name),
                      data_type=clean_xml_string(data_type))

            # 表-列边
            G.add_edge(table_node_id, column_node_id, edge_type='table-column')

    # 2. 添加外键边（增强信息）
    for fk in schema_info.get('foreign_keys', []):
        source_table = fk['table']
        source_column = fk['column']
        target_table = fk['references']['table']
        target_column = fk['references']['column']

        source_node_original = f"{source_table}.{source_column}"
        target_node_original = f"{target_table}.{target_column}"
        
        # 使用清理后的节点ID
        source_node = clean_node_id(source_node_original)
        target_node = clean_node_id(target_node_original)

        if source_node in G.nodes and target_node in G.nodes:
            # 添加外键边，包含关系信息
            G.add_edge(source_node, target_node, 
                      edge_type='foreign-key',
                      source_table=clean_xml_string(source_table),
                      target_table=clean_xml_string(target_table),
                      relationship='references')

    # 3. 解析SQL框架，添加SQL占位符节点
    sql_placeholders = parse_sql_framework(sql_framework)
    for idx, placeholder in enumerate(sql_placeholders):
        placeholder_type = placeholder.get('clause', 'UNKNOWN')
        placeholder_text = placeholder.get('text', '_')
        sql_node = f"sql_placeholder_{idx}"  # 这个ID已经是安全的
        
        G.add_node(sql_node, 
                  node_type='sql_placeholder', 
                  placeholder_type=clean_xml_string(placeholder_type),
                  placeholder_text=clean_xml_string(placeholder_text),
                  position=placeholder.get('position', idx),
                  context=clean_xml_string(placeholder.get('context', '')))

        # 根据占位符类型，连接到可能的模式节点
        possible_nodes = get_possible_schema_nodes(G, placeholder_type)
        for node in possible_nodes:
            G.add_edge(sql_node, node, edge_type='sql-schema')
    
    # 注意：node_id_map不能直接保存到GraphML中（不支持字典类型）
    # 如果需要，可以保存到单独的JSON文件中
    # 目前通过节点的original_name属性可以恢复原始名称

    return G, node_id_map

def extract_graph_metadata(G):
    """
    从图中提取元数据信息，用于后续的智能推理。
    返回：
    - foreign_key_relations: 外键关系列表
    - table_info: 表信息字典
    - column_info: 列信息字典
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
    
    # 提取外键关系
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
    """保存图为GraphML格式"""
    nx.write_graphml(G, output_file)
    # 不打印每个文件的保存信息，减少输出

def save_graph_metadata(metadata, output_file):
    """保存图的元数据为JSON格式"""
    # 将set转换为list以便JSON序列化
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
    
    # 添加node_id_map（如果存在）
    if 'node_id_map' in metadata:
        serializable_metadata['node_id_map'] = metadata['node_id_map']
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_metadata, f, ensure_ascii=False, indent=2)
    # 不打印每个文件的保存信息，减少输出

def process_database(database_name, skeleton_file, schema_file, output_dir):
    """
    处理单个数据库，生成图文件。
    """
    # 加载schema
    if not os.path.exists(schema_file):
        print(f"Schema文件不存在: {schema_file}")
        return
    
    schema = load_schema(schema_file)
    
    # 加载SQL skeletons
    if not os.path.exists(skeleton_file):
        print(f"SQL skeleton文件不存在: {skeleton_file}")
        return
    
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        sql_skeletons = json.load(f)
    
    # 创建输出目录
    graph_dir = os.path.join(output_dir, database_name)
    os.makedirs(graph_dir, exist_ok=True)
    
    print(f"正在处理数据库 '{database_name}'，共 {len(sql_skeletons)} 个SQL骨架...")
    
    # 为每个SQL骨架构建图
    for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} 处理进度", leave=False)):
        # 如果是字符串，直接使用；如果是字典，提取sql_framework字段
        if isinstance(sql_skeleton, dict):
            sql_framework = sql_skeleton.get('sql_framework', '')
        else:
            sql_framework = sql_skeleton
        
        if not sql_framework:
            continue
        
        # 构建图
        G, node_id_map = build_sql_schema_graph(schema, sql_framework)
        
        # 提取元数据
        metadata = extract_graph_metadata(G)
        
        # 将node_id_map添加到元数据中
        metadata['node_id_map'] = node_id_map
        
        # 保存图
        graph_file = os.path.join(graph_dir, f"{database_name}_graph_{idx}.graphml")
        save_graph(G, graph_file)
        
        # 保存元数据（包含node_id_map）
        metadata_file = os.path.join(graph_dir, f"{database_name}_metadata_{idx}.json")
        save_graph_metadata(metadata, metadata_file)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='构建SQL-Schema Linking Graph（改进版）')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    parser.add_argument('--skeleton_dir', type=str, default=None,
                       help='SQL骨架目录（默认：../../data/beijing/output/sql_skeleton）')
    parser.add_argument('--database_dir', type=str, default=None,
                       help='数据库目录（默认：../../data/beijing/database）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：../../data/beijing/output/graph）')
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.skeleton_dir is None:
        args.skeleton_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'sql_skeleton')
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output', 'graph')
    
    # 转换为绝对路径
    args.skeleton_dir = os.path.abspath(args.skeleton_dir)
    args.database_dir = os.path.abspath(args.database_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有SQL skeleton文件
    skeleton_files = [f for f in os.listdir(args.skeleton_dir) if f.endswith('_sql_skeleton.json')]
    
    print(f"找到 {len(skeleton_files)} 个数据库的SQL骨架文件")
    
    for skeleton_file in tqdm(skeleton_files, desc="总体进度"):
        # 提取数据库名称
        database_name = skeleton_file.replace('_sql_skeleton.json', '')
        
        skeleton_path = os.path.join(args.skeleton_dir, skeleton_file)
        schema_path = os.path.join(args.database_dir, database_name, f"{database_name}.json")
        
        process_database(database_name, skeleton_path, schema_path, args.output_dir)
    
    print(f"\n{'='*60}")
    print("✓ 所有数据库的图构建完成！")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

