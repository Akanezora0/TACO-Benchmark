#!/usr/bin/env python3
"""
从图文件中提取关键信息，用于指导SQL骨架填充

核心思路：
1. 分析SQL骨架，确定需要哪些类型的节点
2. 从图中提取与SQL占位符直接相关的节点
3. 提取这些节点之间的外键关系
4. 只包含相关的表和列信息，大幅减小prompt大小
"""

import networkx as nx
import json
import re
from collections import defaultdict

def analyze_sql_framework(sql_framework):
    """
    分析SQL骨架，确定需要哪些类型的表和列
    返回：需要的节点类型和数量
    """
    sql_upper = sql_framework.upper()
    
    # 分析需要的表数量
    has_join = 'JOIN' in sql_upper
    has_union = 'UNION' in sql_upper
    has_subquery = '(' in sql_framework and 'SELECT' in sql_upper
    
    # 统计占位符数量
    placeholder_count = sql_framework.count('_')
    
    # 估计需要的表数量
    if has_join:
        # JOIN通常需要2-3张表
        num_joins = sql_upper.count('JOIN')
        estimated_tables = min(num_joins + 1, 5)
    elif has_union:
        # UNION通常需要2张表
        estimated_tables = 2
    elif has_subquery:
        # 子查询可能需要2-3张表
        estimated_tables = min(3, placeholder_count // 3)
    else:
        # 单表查询
        estimated_tables = 1
    
    # 分析需要的列类型
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
    从图文件中提取与SQL骨架相关的关键节点
    
    策略：
    1. 找到所有SQL占位符节点
    2. 找到与占位符直接连接的表和列节点
    3. 根据SQL骨架语义，选择最相关的表
    4. 提取这些表的列和外键关系
    """
    if G is None:
        return None
    
    # 分析SQL骨架
    sql_analysis = analyze_sql_framework(sql_framework)
    
    # 1. 找到所有SQL占位符节点
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
    
    # 2. 找到与占位符直接连接的所有节点
    connected_nodes = set()
    placeholder_connections = defaultdict(list)
    
    for placeholder in sql_placeholders:
        placeholder_id = placeholder['node_id']
        neighbors = list(G.neighbors(placeholder_id))
        connected_nodes.update(neighbors)
        placeholder_connections[placeholder_id] = neighbors
    
    # 3. 从连接的节点中提取表和列
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
    
    # 4. 如果表太多，根据外键关系优先选择
    if len(relevant_tables) > max_tables:
        # 提取外键关系，优先选择有外键关系的表
        fk_relations = []
        for u, v, attr in G.edges(data=True):
            if attr.get('edge_type') == 'foreign-key':
                source_table = attr.get('source_table', '')
                target_table = attr.get('target_table', '')
                if source_table in relevant_tables and target_table in relevant_tables:
                    fk_relations.append((source_table, target_table))
        
        # 构建表关系图
        table_graph = defaultdict(set)
        for source, target in fk_relations:
            table_graph[source].add(target)
            table_graph[target].add(source)
        
        # 优先选择有外键关系的表
        prioritized_tables = set()
        for source, target in fk_relations[:max_tables]:
            prioritized_tables.add(source)
            prioritized_tables.add(target)
            if len(prioritized_tables) >= max_tables:
                break
        
        # 如果还不够，随机添加其他表
        if len(prioritized_tables) < max_tables:
            remaining = list(relevant_tables - prioritized_tables)
            prioritized_tables.update(remaining[:max_tables - len(prioritized_tables)])
        
        relevant_tables = prioritized_tables
    
    # 5. 限制每张表的列数量
    filtered_table_to_columns = {}
    for table in relevant_tables:
        columns = list(table_to_columns.get(table, []))[:max_columns_per_table]
        if columns:
            filtered_table_to_columns[table] = columns
    
    # 6. 提取外键关系（只包含相关表的）
    foreign_keys = []
    for u, v, attr in G.edges(data=True):
        if attr.get('edge_type') == 'foreign-key':
            source_table = attr.get('source_table', '')
            target_table = attr.get('target_table', '')
            if source_table in relevant_tables and target_table in relevant_tables:
                # 从节点属性中获取列信息
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
    
    # 7. 提取表信息（描述、注释等）
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
    
    # 8. 提取列信息（数据类型等）
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
    将提取的信息格式化成适合放入prompt的文本
    """
    if not extracted_info:
        return ""
    
    tables = extracted_info.get('tables', [])
    table_info = extracted_info.get('table_info', {})
    columns = extracted_info.get('columns', {})
    column_info = extracted_info.get('column_info', {})
    foreign_keys = extracted_info.get('foreign_keys', [])
    
    prompt_text = "\n=== 数据库Schema信息（精简版）===\n\n"
    
    # 表信息
    prompt_text += f"可用表（共{len(tables)}个）：\n"
    for table in tables:
        prompt_text += f"- {table}\n"
    
    # 表详细信息
    if table_info:
        prompt_text += "\n表详细信息：\n"
        for table_name in tables:
            if table_name in table_info:
                info = table_info[table_name]
                prompt_text += f"\n表名：{table_name}\n"
                if info.get('description') and info['description'] != 'No description available.':
                    prompt_text += f"描述：{info['description']}\n"
                if info.get('comment'):
                    prompt_text += f"注释：{info['comment']}\n"
                
                # 列信息
                if table_name in columns:
                    prompt_text += "列信息：\n"
                    for col_name in columns[table_name]:
                        if col_name in column_info:
                            col_info = column_info[col_name]
                            data_type = col_info.get('data_type', 'TEXT')
                            prompt_text += f"  - {col_name} (类型: {data_type})\n"
    
    # 外键关系
    if foreign_keys:
        prompt_text += "\n外键关系（可用于JOIN）：\n"
        for fk in foreign_keys:
            prompt_text += f"- {fk['source_table']}.{fk['source_column']} → {fk['target_table']}.{fk['target_column']}\n"
    
    return prompt_text

def extract_and_save_relevant_info(graph_file, sql_framework, output_file=None):
    """
    从图文件中提取关键信息并保存
    """
    # 加载图文件
    try:
        G = nx.read_graphml(graph_file)
    except Exception as e:
        print(f"加载图文件失败: {e}")
        return None
    
    # 提取关键信息
    extracted_info = extract_relevant_nodes_from_graph(G, sql_framework)
    
    if not extracted_info:
        return None
    
    # 添加SQL骨架信息
    extracted_info['sql_framework'] = sql_framework
    
    # 保存到文件（如果指定了输出文件）
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_info, f, ensure_ascii=False, indent=2)
    
    return extracted_info

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("用法: python3 graph_extractor.py <graph_file> <sql_framework> [output_file]")
        sys.exit(1)
    
    graph_file = sys.argv[1]
    sql_framework = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    extracted_info = extract_and_save_relevant_info(graph_file, sql_framework, output_file)
    
    if extracted_info:
        print("提取成功！")
        print(f"相关表数量: {len(extracted_info['tables'])}")
        print(f"外键关系数量: {len(extracted_info['foreign_keys'])}")
        
        # 格式化输出
        formatted = format_extracted_info_for_prompt(extracted_info, sql_framework)
        print("\n格式化后的信息:")
        print(formatted)
    else:
        print("提取失败")

