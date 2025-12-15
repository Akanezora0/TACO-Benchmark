#!/usr/bin/env python3
"""
提取数据库schema（使用中文表名版本）

直接从数据库JSON文件或SQLite数据库提取标准格式的schema，使用原始中文表名。
"""

import os
import json
import sqlite3
import argparse
from tqdm import tqdm

def extract_schema_from_json(json_file):
    """从数据库JSON文件中提取schema"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    schema = {'tables': []}
    
    # 数据库JSON格式：{表名: {columns: [...], data: [...]}}
    for table_name, table_data in data.items():
        columns = []
        if 'columns' in table_data:
            for col_name in table_data['columns']:
                columns.append({
                    'column_name': col_name,
                    'data_type': 'TEXT'  # 默认类型，可以根据需要推断
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

def extract_schema_from_db(db_file):
    """从SQLite数据库文件中提取schema"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    schema = {'tables': []}
    
    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for (table_name,) in tables:
        # 获取表的列信息
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns_info = cursor.fetchall()
        
        columns = []
        for col_info in columns_info:
            col_name = col_info[1]
            col_type = col_info[2]  # SQLite类型
            columns.append({
                'column_name': col_name,
                'data_type': col_type
            })
        
        schema['tables'].append({
            'table_name': table_name,
            'table_comment': table_name,
            'table_description': 'No description available.',
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        })
    
    conn.close()
    return schema

def process_databases(database_dir, output_dir):
    """处理所有数据库，提取schema"""
    # 获取所有数据库文件夹
    db_folders = [f for f in os.listdir(database_dir) 
                  if os.path.isdir(os.path.join(database_dir, f))]
    
    print(f"找到 {len(db_folders)} 个数据库")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for db_name in tqdm(db_folders, desc="提取schema"):
        db_folder = os.path.join(database_dir, db_name)
        
        # 尝试从JSON文件提取
        json_file = os.path.join(db_folder, f"{db_name}.json")
        db_file = os.path.join(db_folder, f"{db_name}.db")
        
        schema = None
        if os.path.exists(json_file):
            try:
                schema = extract_schema_from_json(json_file)
            except Exception as e:
                print(f"\n从JSON提取schema失败 {json_file}: {e}")
        
        if schema is None and os.path.exists(db_file):
            try:
                schema = extract_schema_from_db(db_file)
            except Exception as e:
                print(f"\n从数据库提取schema失败 {db_file}: {e}")
        
        if schema:
            # 保存schema文件
            schema_file = os.path.join(output_dir, f"{db_name}_schema.json")
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(schema, f, ensure_ascii=False, indent=2)
        else:
            print(f"\n警告：无法提取数据库 '{db_name}' 的schema")

def main():
    parser = argparse.ArgumentParser(description='提取数据库schema（使用中文表名）')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    parser.add_argument('--database_dir', type=str, default=None,
                       help='数据库目录（默认：../../data/beijing/database_chinese）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='schema输出目录（默认：../../data/beijing/schema_chinese）')
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database_chinese')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'schema_chinese')
    
    # 转换为绝对路径
    args.database_dir = os.path.abspath(args.database_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    
    print(f"{'='*60}")
    print("提取数据库schema（使用中文表名）")
    print(f"{'='*60}")
    print(f"数据库目录: {args.database_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")
    
    process_databases(args.database_dir, args.output_dir)
    
    print(f"\n{'='*60}")
    print("✓ 所有schema提取完成！")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

