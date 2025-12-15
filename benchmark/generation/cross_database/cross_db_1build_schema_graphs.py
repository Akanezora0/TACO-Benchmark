#!/usr/bin/env python3
"""
跨数据库SQL-Schema Linking Graph构建脚本

基于单数据库的图生成脚本，扩展支持跨数据库场景：
1. 加载多个数据库的schema
2. 为跨数据库SQL骨架生成图
3. 考虑跨数据库的表和列关系
"""

import json
import networkx as nx
import re
import os
import importlib.util
from tqdm import tqdm
from collections import defaultdict
import argparse

# 导入单数据库的图生成函数
import sys
sql_filling_dir = os.path.join(os.path.dirname(__file__), '..', 'sql_filling')
sys.path.insert(0, sql_filling_dir)

# 动态导入
import importlib.util
spec = importlib.util.spec_from_file_location(
    "build_schema_graphs_improved",
    os.path.join(sql_filling_dir, "1build_schema_graphs_improved.py")
)
build_graphs_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_graphs_module)

load_schema = build_graphs_module.load_schema
parse_sql_framework = build_graphs_module.parse_sql_framework
get_possible_schema_nodes = build_graphs_module.get_possible_schema_nodes

def load_multiple_schemas(database_names, database_dir):
    """加载多个数据库的schema信息"""
    schemas = {}
    for db_name in database_names:
        schema_file = os.path.join(database_dir, db_name, f"{db_name}.json")
        if os.path.exists(schema_file):
            schemas[db_name] = load_schema(schema_file)
        else:
            print(f"警告: 找不到schema文件 {schema_file}")
    return schemas

def build_cross_database_graph(sql_skeleton, schemas, table_database_mapping):
    """
    为跨数据库SQL骨架构建图
    关键：同时加载所有涉及的数据库的schema，构建统一的图
    
    Args:
        sql_skeleton: SQL骨架字符串
        schemas: 多个数据库的schema字典 {db_name: schema}
        table_database_mapping: 表到数据库的映射 {table_name: db_name}
    
    Returns:
        NetworkX图对象和元数据
    """
    G = nx.DiGraph()
    
    # 1. 添加所有涉及的数据库的表和列节点（统一构建）
    # 这是关键：不是分别建图，而是将所有数据库的表和列都添加到同一个图中
    for db_name, schema in schemas.items():
        if schema is None:
            continue
            
        for table_info in schema.get('tables', []):
            table_name = table_info.get('table_name', '')
            # 节点ID格式：数据库名.表名（确保跨数据库唯一性）
            table_node_id = f"{db_name}.{table_name}"
            
            # 添加表节点
            G.add_node(table_node_id, 
                      node_type='table',
                      table_name=table_name,
                      database=db_name,
                      table_comment=table_info.get('table_comment', ''),
                      table_description=table_info.get('table_description', ''))
            
            # 添加列节点
            for col_info in table_info.get('columns', []):
                col_name = col_info.get('column_name', '')
                # 节点ID格式：数据库名.表名.列名
                col_node_id = f"{db_name}.{table_name}.{col_name}"
                
                G.add_node(col_node_id,
                          node_type='column',
                          column_name=col_name,
                          table_name=table_name,
                          database=db_name,
                          data_type=col_info.get('data_type', 'TEXT'))
                
                # 添加表到列的边
                G.add_edge(table_node_id, col_node_id, edge_type='contains')
    
    # 2. 添加外键关系（跨数据库的外键关系）
    for db_name, schema in schemas.items():
        for table_info in schema.get('tables', []):
            table_name = table_info.get('table_name', '')
            table_node_id = f"{db_name}.{table_name}"
            
            # 处理外键关系
            for fk in table_info.get('foreign_keys', []):
                ref_table = fk.get('referenced_table', '')
                ref_col = fk.get('referenced_column', '')
                fk_col = fk.get('column_name', '')
                
                # 查找引用表所在的数据库
                ref_db = table_database_mapping.get(ref_table, db_name)  # 默认同数据库
                
                if ref_table and ref_col:
                    source_col_id = f"{db_name}.{table_name}.{fk_col}"
                    target_col_id = f"{ref_db}.{ref_table}.{ref_col}"
                    
                    # 如果目标列节点存在，添加外键边
                    if G.has_node(target_col_id):
                        G.add_edge(source_col_id, target_col_id, edge_type='foreign_key')
    
    # 3. 解析SQL骨架，识别占位符
    placeholders = parse_sql_framework(sql_skeleton)
    
    # 4. 为每个占位符添加可能的连接
    placeholder_nodes = {}
    for i, placeholder in enumerate(placeholders):
        placeholder_id = f"placeholder_{i}"
        placeholder_nodes[placeholder_id] = placeholder
        
        # 根据占位符类型，获取可能的schema节点
        # 注意：这里需要考虑跨数据库的情况
        possible_nodes = get_possible_schema_nodes(G, placeholder['clause'])
        
        # 过滤：只考虑table_database_mapping中涉及的表
        relevant_tables = set(table_database_mapping.keys())
        filtered_nodes = []
        for node_id in possible_nodes:
            # 提取表名（从节点ID中）
            if '.' in node_id:
                parts = node_id.split('.')
                if len(parts) >= 2:
                    table_name = parts[1]  # 格式：db.table 或 db.table.col
                    if table_name in relevant_tables:
                        filtered_nodes.append(node_id)
        
        # 添加占位符节点
        G.add_node(placeholder_id,
                  node_type='placeholder',
                  placeholder_type=placeholder['clause'],
                  position=placeholder['position'])
        
        # 连接到可能的schema节点
        for node_id in filtered_nodes:
            G.add_edge(placeholder_id, node_id, edge_type='possible_match')
    
    # 5. 构建元数据
    metadata = {
        'databases': list(schemas.keys()),
        'table_database_mapping': table_database_mapping,
        'placeholders': placeholders,
        'num_tables': len(set(table_database_mapping.keys())),
        'num_databases': len(schemas)
    }
    
    return G, metadata

def process_cross_database_skeleton(skeleton_data, schemas, output_dir):
    """处理单个跨数据库SQL骨架，生成图文件"""
    sql_skeleton = skeleton_data['sql_skeleton']
    table_database_mapping = skeleton_data['table_database_mapping']
    
    # 构建图
    G, metadata = build_cross_database_graph(
        sql_skeleton, 
        schemas, 
        table_database_mapping
    )
    
    # 保存图（使用NetworkX的JSON格式）
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
    
    # 确定输出文件名（包含组合信息）
    original_file = skeleton_data.get('original_file', 'unknown')
    databases = skeleton_data.get('databases', [])
    combo_name = '_'.join(sorted(databases)) if databases else 'unknown'
    
    # 从 original_file 中提取索引（如 generated_sql_1.json -> 1 或 join_skeleton_0 -> 0）
    match = re.search(r'(\d+)', original_file)
    if match:
        idx = match.group(1)
        output_file = os.path.join(output_dir, f"cross_db_graph_{idx}.json")
    else:
        # 如果没有找到索引，使用hash
        import hashlib
        hash_id = hashlib.md5(sql_skeleton.encode()).hexdigest()[:8]
        output_file = os.path.join(output_dir, f"cross_db_graph_{hash_id}.json")
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='为跨数据库SQL骨架生成图')
    parser.add_argument('--skeleton_file', type=str, required=True,
                       help='跨数据库SQL骨架文件')
    parser.add_argument('--database_dir', type=str,
                       default='benchmark/data/beijing/database_chinese',
                       help='数据库目录')
    parser.add_argument('--output_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_graph',
                       help='图文件输出目录')
    
    args = parser.parse_args()
    
    # 加载跨数据库SQL骨架
    print(f"加载跨数据库SQL骨架: {args.skeleton_file}")
    with open(args.skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    
    print(f"共 {len(skeletons)} 个SQL骨架")
    
    # 获取所有涉及的数据库
    all_databases = set()
    for skeleton in skeletons:
        all_databases.update(skeleton.get('databases', []))
    
    print(f"涉及的数据库: {sorted(all_databases)}")
    
    # 加载所有数据库的schema
    print("\n加载数据库schema...")
    schemas = load_multiple_schemas(all_databases, args.database_dir)
    print(f"成功加载 {len(schemas)} 个数据库的schema")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个SQL骨架
    print(f"\n生成图文件...")
    success_count = 0
    for skeleton in tqdm(skeletons, desc="生成图"):
        try:
            output_file = process_cross_database_skeleton(
                skeleton, 
                schemas, 
                args.output_dir
            )
            success_count += 1
        except Exception as e:
            print(f"\n处理骨架失败: {e}")
            continue
    
    print(f"\n完成！成功生成 {success_count}/{len(skeletons)} 个图文件")
    print(f"输出目录: {args.output_dir}")

if __name__ == '__main__':
    main()

