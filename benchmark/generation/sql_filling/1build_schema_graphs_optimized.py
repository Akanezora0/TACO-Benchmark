#!/usr/bin/env python3
"""
优化的SQL-Schema Linking Graph构建脚本

优化点：
1. 不保存完整的GraphML文件，只保存精简的JSON格式元数据
2. 元数据只包含必要的表、列、外键关系信息
3. 大幅减小文件大小，方便放入prompt
"""

import json
import os
from tqdm import tqdm

def load_schema(schema_file):
    """加载数据库的schema信息"""
    with open(schema_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否是标准schema格式（包含'tables'键）
    if 'tables' in data:
        return data
    
    # 如果不是标准格式，从数据库JSON文件中提取schema
    schema = {'tables': []}
    for table_name, table_data in data.items():
        columns = []
        if 'columns' in table_data:
            for col_name in table_data['columns']:
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'
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

def extract_schema_metadata(schema_info):
    """
    从schema中提取精简的元数据信息
    只包含必要的表、列、外键关系信息
    """
    metadata = {
        'tables': {},
        'foreign_keys': []
    }
    
    # 提取表信息
    for table in schema_info['tables']:
        table_name = table['table_name']
        columns = []
        for column in table['columns']:
            columns.append({
                'name': column['column_name'],
                'data_type': column.get('data_type', 'TEXT')
            })
        
        metadata['tables'][table_name] = {
            'name': table_name,
            'comment': table.get('table_comment', ''),
            'description': table.get('table_description', 'No description available.'),
            'columns': columns
        }
    
    # 提取外键关系
    for table in schema_info['tables']:
        for fk in table.get('foreign_keys', []):
            metadata['foreign_keys'].append({
                'source_table': fk.get('table', table['table_name']),
                'source_column': fk.get('column', ''),
                'target_table': fk.get('references', {}).get('table', ''),
                'target_column': fk.get('references', {}).get('column', '')
            })
    
    return metadata

def save_metadata(metadata, output_file):
    """保存精简的元数据为JSON格式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def process_database(database_name, skeleton_file, schema_file, output_dir):
    """
    处理单个数据库，生成精简的元数据文件
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
    
    # 提取schema元数据（所有SQL骨架共享同一个schema）
    schema_metadata = extract_schema_metadata(schema)
    
    # 为每个SQL骨架保存元数据（实际上所有骨架共享同一个schema，但为了兼容性，每个都保存一份）
    for idx, sql_skeleton in enumerate(tqdm(sql_skeletons, desc=f"{database_name} 处理进度", leave=False)):
        # 如果是字典，提取sql_framework字段
        if isinstance(sql_skeleton, dict):
            sql_framework = sql_skeleton.get('sql_framework', '')
        else:
            sql_framework = sql_skeleton
        
        if not sql_framework:
            continue
        
        # 创建包含SQL骨架信息的元数据
        metadata = {
            'sql_framework': sql_framework,
            'database_name': database_name,
            'skeleton_index': idx,
            **schema_metadata
        }
        
        # 保存元数据（精简格式，不包含图结构）
        metadata_file = os.path.join(graph_dir, f"{database_name}_metadata_{idx}.json")
        save_metadata(metadata, metadata_file)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='构建SQL-Schema元数据（优化版）')
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
    print("✓ 所有数据库的元数据构建完成！")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

