#!/usr/bin/env python3
"""
使用中文表名创建数据库（优化版）

关键改进：
1. 直接使用原始CSV文件名（去掉.csv后缀）作为表名，不进行拼音转换
2. 所有表名和列名都用双引号包裹，确保SQLite可以正确处理中文和特殊字符
3. 生成标准schema格式的JSON文件，便于后续使用
"""

import os
import sqlite3
import pandas as pd
import json
from tqdm import tqdm

def quote_identifier(identifier):
    """使用双引号包裹标识符，确保SQLite可以正确处理中文和特殊字符"""
    # 转义双引号
    escaped = str(identifier).replace('"', '""')
    return f'"{escaped}"'

def csv_to_sqlite_chinese(csv_folder_path, sqlite_db_path):
    """
    将 CSV 文件转换为 SQLite 表，使用中文表名
    
    关键改进：
    - 直接使用CSV文件名（去掉.csv后缀）作为表名，不进行拼音转换
    - 使用双引号包裹表名和列名，确保SQLite可以正确处理中文
    """
    # 创建 SQLite 数据库连接
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    # 存储数据库结构信息
    db_structure = {}
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
    
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 遍历 CSV 文件，将每个文件转化为 SQLite 表
    for file_name in tqdm(csv_files, desc="处理CSV文件"):
        csv_path = os.path.join(csv_folder_path, file_name)
        try:
            # 读取 CSV 文件
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # 将 NaN 替换为空字符串
            df = df.fillna('')

            # 获取表名（去掉 .csv 后缀，使用原始中文名称）
            table_name = file_name.replace('.csv', '')
            
            # 使用双引号包裹表名，确保SQLite可以正确处理中文
            quoted_table_name = quote_identifier(table_name)
            
            # 手动创建表（使用双引号包裹表名和列名）
            # 首先删除已存在的表
            cursor.execute(f'DROP TABLE IF EXISTS {quoted_table_name}')
            
            # 构建CREATE TABLE语句
            column_defs = []
            for col in df.columns:
                # 推断数据类型
                col_type = 'TEXT'  # 默认类型
                if df[col].dtype == 'int64':
                    col_type = 'INTEGER'
                elif df[col].dtype == 'float64':
                    col_type = 'REAL'
                
                quoted_col = quote_identifier(col)
                column_defs.append(f'{quoted_col} {col_type}')
            
            create_table_sql = f'CREATE TABLE {quoted_table_name} ({", ".join(column_defs)})'
            cursor.execute(create_table_sql)
            
            # 插入数据
            for _, row in df.iterrows():
                # 构建INSERT语句，使用双引号包裹表名和列名
                columns = ', '.join([quote_identifier(col) for col in df.columns])
                placeholders = ', '.join(['?' for _ in df.columns])
                insert_sql = f'INSERT INTO {quoted_table_name} ({columns}) VALUES ({placeholders})'
                cursor.execute(insert_sql, tuple(row))
            
            # 获取表的列名并存储到 db_structure
            db_structure[table_name] = {
                'columns': df.columns.tolist(),
                'row_count': len(df)
            }
            
        except Exception as e:
            # 如果出现错误，跳过该文件并打印错误信息
            print(f"\n错误：处理文件 {csv_path} 时出错: {e}")
            continue  # 跳过当前文件，继续处理下一个文件

    # 提交并关闭数据库连接
    conn.commit()
    conn.close()

    # 返回数据库结构
    return db_structure

def save_schema_as_json(db_structure, db_folder_path, db_name):
    """
    保存标准格式的schema JSON文件
    
    格式：
    {
        "tables": [
            {
                "table_name": "表名",
                "table_comment": "表名",
                "table_description": "No description available.",
                "columns": [
                    {
                        "column_name": "列名",
                        "data_type": "TEXT"
                    }
                ],
                "primary_keys": [],
                "foreign_keys": []
            }
        ]
    }
    """
    schema = {'tables': []}
    
    for table_name, table_info in db_structure.items():
        columns = []
        for col_name in table_info['columns']:
            # 默认类型为TEXT，实际使用时可以根据需要推断
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
    
    # 保存schema JSON文件
    schema_file_path = os.path.join(db_folder_path, f"{db_name}.json")
    try:
        with open(schema_file_path, 'w', encoding='utf-8') as f:
            json.dump(schema, f, ensure_ascii=False, indent=2)
        print(f"Schema已保存: {schema_file_path}")
    except Exception as e:
        print(f"保存schema文件时出错 {schema_file_path}: {e}")

def process_parsed_data_to_sqlite_chinese(parsed_data_dir, database_dir):
    """
    处理 parsed_data 中的所有文件夹，生成使用中文表名的 SQLite 数据库
    """
    # 遍历 parsed_data/ 下的所有文件夹
    folders = [f for f in os.listdir(parsed_data_dir) 
               if os.path.isdir(os.path.join(parsed_data_dir, f))]
    
    print(f"找到 {len(folders)} 个数据库文件夹")
    
    for folder_name in tqdm(folders, desc="处理数据库"):
        folder_path = os.path.join(parsed_data_dir, folder_name)
        try:
            # 为每个文件夹创建一个 SQLite 数据库
            db_folder_path = os.path.join(database_dir, folder_name)
            os.makedirs(db_folder_path, exist_ok=True)
            
            sqlite_db_path = os.path.join(db_folder_path, f"{folder_name}.db")
            
            # 转化该文件夹中的所有 CSV 为 SQLite 表，使用中文表名
            db_structure = csv_to_sqlite_chinese(folder_path, sqlite_db_path)
            
            # 保存标准格式的schema JSON文件
            save_schema_as_json(db_structure, db_folder_path, folder_name)
            
            print(f"\n✓ 数据库 '{folder_name}' 创建完成")
            print(f"  表数量: {len(db_structure)}")
            print(f"  数据库文件: {sqlite_db_path}")
            
        except Exception as e:
            # 如果处理文件夹时出错，打印错误信息并继续处理下一个文件夹
            print(f"\n✗ 处理文件夹 {folder_path} 时出错: {e}")
            continue  # 跳过当前文件夹，继续处理下一个文件夹

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='使用中文表名创建SQLite数据库')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    parser.add_argument('--parsed_data_dir', type=str, default=None,
                       help='parsed_data目录（默认：../../data/parsed_data）')
    parser.add_argument('--database_dir', type=str, default=None,
                       help='数据库输出目录（默认：../../data/database_chinese）')
    
    args = parser.parse_args()
    
    # 设置默认路径
    if args.parsed_data_dir is None:
        args.parsed_data_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'parsed_data')
    if args.database_dir is None:
        args.database_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'database_chinese')
    
    # 转换为绝对路径
    args.parsed_data_dir = os.path.abspath(args.parsed_data_dir)
    args.database_dir = os.path.abspath(args.database_dir)
    
    # 创建输出目录
    os.makedirs(args.database_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("使用中文表名创建SQLite数据库")
    print(f"{'='*60}")
    print(f"输入目录: {args.parsed_data_dir}")
    print(f"输出目录: {args.database_dir}")
    print(f"{'='*60}\n")
    
    # 处理 parsed_data 中的所有文件夹，生成 SQLite 数据库
    process_parsed_data_to_sqlite_chinese(args.parsed_data_dir, args.database_dir)
    
    print(f"\n{'='*60}")
    print("✓ 所有数据库创建完成！")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

