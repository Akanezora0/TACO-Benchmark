#!/usr/bin/env python3
"""
准备US数据集，按照beijing数据集的流程处理

步骤：
1. 创建US数据集的目录结构
2. 转换数据库格式（从old格式转换为标准格式）
3. 提取SQL骨架并按数据库分组
4. 准备运行图生成和SQL填充
"""

import os
import json
import shutil
import sqlite3
from pathlib import Path

def convert_us_database_format(old_db_path, old_json_path, new_db_dir, db_name):
    """
    转换US数据集的数据库格式
    
    从old格式（顶层键是表名）转换为标准格式（有tables键）
    """
    # 读取old格式的JSON
    with open(old_json_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    # 转换为标准格式
    schema = {'tables': []}
    
    for table_name, table_data in old_data.items():
        if isinstance(table_data, dict) and 'columns' in table_data:
            columns = []
            for col_name in table_data['columns']:
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
    
    # 创建新目录
    os.makedirs(new_db_dir, exist_ok=True)
    
    # 保存标准格式的schema JSON
    schema_file = os.path.join(new_db_dir, f"{db_name}.json")
    with open(schema_file, 'w', encoding='utf-8') as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    
    # 复制数据库文件
    new_db_file = os.path.join(new_db_dir, f"{db_name}.db")
    if os.path.exists(old_db_path):
        shutil.copy2(old_db_path, new_db_file)
    
    return schema_file, new_db_file

def extract_sql_skeletons_by_database(skeleton_file, output_dir):
    """
    从new_sql_skeletons.json中提取SQL骨架，按数据库分组
    
    需要从SQL中推断数据库名（通过表名）
    """
    with open(skeleton_file, 'r', encoding='utf-8') as f:
        skeletons = json.load(f)
    
    # 从SQL中提取表名，推断数据库
    # 这里需要根据实际SQL格式来推断
    # 暂时先按表名分组
    
    db_skeletons = {}
    
    for item in skeletons:
        if isinstance(item, dict):
            sql = item.get('sql', '')
            sql_framework = item.get('sql_framework', '')
            
            # 从SQL中提取表名（FROM后的第一个标识符）
            if 'FROM' in sql.upper():
                parts = sql.upper().split('FROM')
                if len(parts) > 1:
                    table_part = parts[1].strip().split()[0].strip(';')
                    table_name = table_part.strip()
                    
                    # 根据表名推断数据库（这里需要建立表名到数据库的映射）
                    # 暂时先使用表名作为数据库名（后续需要根据实际数据调整）
                    db_name = table_name  # 临时方案
                    
                    if db_name not in db_skeletons:
                        db_skeletons[db_name] = []
                    
                    db_skeletons[db_name].append({
                        'sql_framework': sql_framework,
                        'sql': sql
                    })
    
    # 保存按数据库分组的SQL骨架
    os.makedirs(output_dir, exist_ok=True)
    
    for db_name, skeletons in db_skeletons.items():
        skeleton_file = os.path.join(output_dir, f"{db_name}_sql_skeleton.json")
        with open(skeleton_file, 'w', encoding='utf-8') as f:
            json.dump(skeletons, f, ensure_ascii=False, indent=2)
    
    return db_skeletons

def main():
    # 设置路径
    project_root = Path(__file__).parent.parent.parent
    old_america_dir = project_root / 'old' / 'saturn' / 'America' / 'data'
    old_us_dir = project_root / 'old' / 'saturn' / 'TACO-Benchmark' / 'us' / 'data'
    new_us_dir = project_root / 'benchmark' / 'data' / 'us'
    
    # 创建新目录结构
    new_db_dir = new_us_dir / 'database_chinese'
    new_skeleton_dir = new_us_dir / 'output' / 'sql_skeleton'
    new_graph_dir = new_us_dir / 'output' / 'graph_chinese'
    new_output_dir = new_us_dir / 'output' / 'single'
    
    for dir_path in [new_db_dir, new_skeleton_dir, new_graph_dir, new_output_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("=== 准备US数据集 ===\n")
    
    # 步骤1: 转换数据库格式
    print("步骤1: 转换数据库格式...")
    old_db_dir = old_america_dir / 'database'
    
    if old_db_dir.exists():
        db_count = 0
        for db_name in os.listdir(old_db_dir):
            db_path = old_db_dir / db_name
            if db_path.is_dir():
                # 查找.db和.json文件
                db_files = list(db_path.glob('*.db'))
                json_files = list(db_path.glob('*.json'))
                
                if db_files and json_files:
                    old_db_file = db_files[0]
                    old_json_file = json_files[0]
                    
                    # 创建新数据库目录（使用安全的目录名）
                    safe_db_name = db_name.replace('/', '_').replace('\\', '_')
                    new_db_subdir = new_db_dir / safe_db_name
                    
                    try:
                        schema_file, new_db_file = convert_us_database_format(
                            str(old_db_file), str(old_json_file),
                            str(new_db_subdir), safe_db_name
                        )
                        db_count += 1
                        print(f"  ✓ {safe_db_name[:60]}...")
                    except Exception as e:
                        print(f"  ✗ {safe_db_name[:60]}... 错误: {e}")
        
        print(f"\n转换完成: {db_count}个数据库\n")
    
    # 步骤2: 提取SQL骨架
    print("步骤2: 提取SQL骨架...")
    old_skeleton_file = old_us_dir.parent / 'new_sql_skeletons.json'
    
    if old_skeleton_file.exists():
        try:
            db_skeletons = extract_sql_skeletons_by_database(
                str(old_skeleton_file), str(new_skeleton_dir)
            )
            print(f"  提取了 {len(db_skeletons)} 个数据库的SQL骨架")
        except Exception as e:
            print(f"  提取失败: {e}")
    
    print("\n=== US数据集准备完成 ===")
    print(f"数据库目录: {new_db_dir}")
    print(f"SQL骨架目录: {new_skeleton_dir}")
    print(f"图文件目录: {new_graph_dir}")
    print(f"输出目录: {new_output_dir}")

if __name__ == '__main__':
    main()

