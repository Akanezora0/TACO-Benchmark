#!/usr/bin/env python3
"""
从现有数据库JSON文件创建使用中文表名的数据库

由于没有parsed_data目录，我们可以：
1. 从现有的数据库JSON文件读取数据
2. 使用table_name_mappings.json反向查找原始中文表名
3. 创建新的使用中文表名的数据库
"""

import os
import sqlite3
import json
from tqdm import tqdm

def quote_identifier(identifier):
    """使用双引号包裹标识符，确保SQLite可以正确处理中文和特殊字符"""
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'

def load_table_name_mappings(mappings_file):
    """加载表名映射，用于反向查找原始中文表名"""
    if not os.path.exists(mappings_file):
        return {}
    
    with open(mappings_file, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    # 创建反向映射：{数据库名: {拼音表名: 原始CSV文件名}}
    reverse_mappings = {}
    for db_name, db_mappings in mappings.items():
        reverse_mappings[db_name] = {v: k for k, v in db_mappings.items()}
    
    return reverse_mappings

def create_database_from_json(json_file, db_file, reverse_mappings, db_name):
    """从JSON文件创建使用中文表名的数据库"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # 获取该数据库的反向映射
    db_reverse_mapping = reverse_mappings.get(db_name, {})
    
    created_tables = 0
    
    for pinyin_table_name, table_data in tqdm(data.items(), desc=f"处理 {db_name}", leave=False):
        # 尝试从反向映射中获取原始中文表名
        original_csv_name = db_reverse_mapping.get(pinyin_table_name, None)
        
        if original_csv_name:
            # 去掉.csv后缀，得到原始表名
            original_table_name = original_csv_name.replace('.csv', '')
        else:
            # 如果没有映射，跳过此表（可能是脏数据或映射不完整）
            print(f"\n警告：表 '{pinyin_table_name}' 在映射中不存在，跳过")
            continue
        
        # 使用双引号包裹表名
        quoted_table_name = quote_identifier(original_table_name)
        
        # 获取列信息
        columns = table_data.get('columns', [])
        if not columns:
            continue
        
        # 构建CREATE TABLE语句
        column_defs = []
        for col in columns:
            quoted_col = quote_identifier(col)
            column_defs.append(f'{quoted_col} TEXT')  # 默认使用TEXT类型
        
        create_table_sql = f'CREATE TABLE {quoted_table_name} ({", ".join(column_defs)})'
        
        try:
            cursor.execute(f'DROP TABLE IF EXISTS {quoted_table_name}')
            cursor.execute(create_table_sql)
            
            # 插入数据
            rows = table_data.get('data', [])
            if rows:
                columns_str = ', '.join([quote_identifier(col) for col in columns])
                placeholders = ', '.join(['?' for _ in columns])
                insert_sql = f'INSERT INTO {quoted_table_name} ({columns_str}) VALUES ({placeholders})'
                
                for row in rows:
                    values = [row.get(col, '') for col in columns]
                    cursor.execute(insert_sql, tuple(values))
            
            created_tables += 1
        except Exception as e:
            print(f"\n错误：创建表 {original_table_name} 时出错: {e}")
            continue
    
    conn.commit()
    conn.close()
    
    return created_tables

def save_schema_as_json(data, db_folder_path, db_name, reverse_mappings):
    """保存标准格式的schema JSON文件"""
    db_reverse_mapping = reverse_mappings.get(db_name, {})
    schema = {'tables': []}
    
    for pinyin_table_name, table_data in data.items():
        # 获取原始中文表名
        original_csv_name = db_reverse_mapping.get(pinyin_table_name, None)
        if original_csv_name:
            original_table_name = original_csv_name.replace('.csv', '')
        else:
            original_table_name = pinyin_table_name
        
        columns = []
        for col_name in table_data.get('columns', []):
            columns.append({
                'column_name': col_name,
                'data_type': 'TEXT'
            })
        
        schema['tables'].append({
            'table_name': original_table_name,
            'table_comment': original_table_name,
            'table_description': 'No description available.',
            'columns': columns,
            'primary_keys': [],
            'foreign_keys': []
        })
    
    schema_file_path = os.path.join(db_folder_path, f"{db_name}.json")
    with open(schema_file_path, 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    
    return schema_file_path

def process_existing_databases(database_dir, output_dir, mappings_file):
    """从现有数据库创建使用中文表名的新数据库"""
    # 加载表名映射
    reverse_mappings = load_table_name_mappings(mappings_file)
    
    # 获取所有数据库文件夹
    db_folders = [f for f in os.listdir(database_dir) 
                  if os.path.isdir(os.path.join(database_dir, f))]
    
    print(f"找到 {len(db_folders)} 个数据库")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for db_name in tqdm(db_folders, desc="处理数据库"):
        db_folder = os.path.join(database_dir, db_name)
        json_file = os.path.join(db_folder, f"{db_name}.json")
        
        if not os.path.exists(json_file):
            print(f"\n警告：跳过 {db_name}，找不到JSON文件")
            continue
        
        try:
            # 读取现有数据库JSON
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 创建输出目录
            output_db_folder = os.path.join(output_dir, db_name)
            os.makedirs(output_db_folder, exist_ok=True)
            
            # 创建数据库文件
            db_file = os.path.join(output_db_folder, f"{db_name}.db")
            created_tables = create_database_from_json(json_file, db_file, reverse_mappings, db_name)
            
            # 保存schema JSON
            schema_file = save_schema_as_json(data, output_db_folder, db_name, reverse_mappings)
            
            print(f"\n✓ {db_name}: 创建了 {created_tables} 个表")
            print(f"  数据库: {db_file}")
            print(f"  Schema: {schema_file}")
            
        except Exception as e:
            print(f"\n✗ 处理 {db_name} 时出错: {e}")
            continue

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='从现有数据库创建使用中文表名的数据库')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    parser.add_argument('--database_dir', type=str, default=None,
                       help='现有数据库目录（默认：../../data/beijing/database）')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（默认：../../data/beijing/database_chinese）')
    parser.add_argument('--mappings_file', type=str, default=None,
                       help='表名映射文件（默认：old/saturn/TACO-Benchmark-all/beijing/data/table_name_mappings.json）')
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database')
    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database_chinese')
    if args.mappings_file is None:
        args.mappings_file = os.path.join(project_root, 'old', 'saturn', 'TACO-Benchmark-all', 'beijing', 'data', 'table_name_mappings.json')
    
    # 转换为绝对路径
    args.database_dir = os.path.abspath(args.database_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    args.mappings_file = os.path.abspath(args.mappings_file)
    
    print(f"{'='*60}")
    print("从现有数据库创建使用中文表名的数据库")
    print(f"{'='*60}")
    print(f"输入目录: {args.database_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"映射文件: {args.mappings_file}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(args.mappings_file):
        print(f"警告：找不到映射文件 {args.mappings_file}")
        print("将使用拼音表名（不推荐）")
    
    process_existing_databases(args.database_dir, args.output_dir, args.mappings_file)
    
    print(f"\n{'='*60}")
    print("✓ 所有数据库创建完成！")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

