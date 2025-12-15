#!/usr/bin/env python3
"""
统计JOIN SQL中涉及2、3、4个数据库的数量
"""

import os
import json
import argparse
from collections import defaultdict

def count_databases_in_sql(data):
    """从SQL数据中提取涉及的数据库数量"""
    # 方法1: 从databases字段获取
    if 'databases' in data:
        databases = data['databases']
        if isinstance(databases, list):
            return len(set(databases))
        elif isinstance(databases, dict):
            return len(databases)
    
    # 方法2: 从table_database_mapping获取
    if 'table_database_mapping' in data:
        mapping = data['table_database_mapping']
        if isinstance(mapping, dict):
            databases = set()
            for table, db in mapping.items():
                if isinstance(db, str):
                    databases.add(db)
                elif isinstance(db, dict) and 'database' in db:
                    databases.add(db['database'])
            return len(databases)
    
    # 方法3: 从schema_graphs获取
    if 'schema_graphs' in data:
        schema_graphs = data['schema_graphs']
        if isinstance(schema_graphs, dict):
            return len(schema_graphs)
    
    return 0

def statistics_join_sqls(backup_dir):
    """统计JOIN SQL的数据库数量分布"""
    
    if not os.path.exists(backup_dir):
        print(f"备份目录不存在: {backup_dir}")
        return
    
    files = [f for f in os.listdir(backup_dir) 
             if f.startswith('cross_db_generated_sql_') and f.endswith('.json')]
    
    print(f"找到 {len(files)} 个SQL文件")
    print()
    
    # 统计
    db_count_stats = defaultdict(int)  # 数据库数量 -> 文件数量
    db_count_details = defaultdict(list)  # 数据库数量 -> 文件列表
    
    valid_files = 0
    invalid_files = 0
    
    for f in files:
        file_path = os.path.join(backup_dir, f)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # 检查是否有结果
                results = data.get('results', [])
                if results is None or len(results) == 0:
                    continue
                
                db_count = count_databases_in_sql(data)
                
                if db_count > 0:
                    db_count_stats[db_count] += 1
                    db_count_details[db_count].append(f)
                    valid_files += 1
                else:
                    invalid_files += 1
                    
        except Exception as e:
            print(f"读取文件失败 {f}: {e}")
            invalid_files += 1
    
    # 输出统计结果
    print("=" * 70)
    print("JOIN SQL 数据库数量统计")
    print("=" * 70)
    print(f"有效文件数: {valid_files}")
    if invalid_files > 0:
        print(f"无效文件数: {invalid_files}")
    print()
    
    print("数据库数量分布:")
    print("-" * 70)
    total = 0
    for db_count in sorted(db_count_stats.keys()):
        count = db_count_stats[db_count]
        total += count
        percentage = (count / valid_files * 100) if valid_files > 0 else 0
        print(f"  {db_count} 个数据库: {count} 个SQL ({percentage:.1f}%)")
    
    print("-" * 70)
    print(f"  总计: {total} 个SQL")
    print()
    
    # 详细列出2、3、4个数据库的文件
    for db_count in [2, 3, 4]:
        if db_count in db_count_details:
            files_list = db_count_details[db_count]
            print(f"{db_count} 个数据库的SQL文件 ({len(files_list)} 个):")
            # 只显示前10个和后10个
            if len(files_list) <= 20:
                for f in files_list:
                    print(f"  - {f}")
            else:
                for f in files_list[:10]:
                    print(f"  - {f}")
                print(f"  ... (省略 {len(files_list) - 20} 个) ...")
                for f in files_list[-10:]:
                    print(f"  - {f}")
            print()
    
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description='统计JOIN SQL中涉及2、3、4个数据库的数量')
    parser.add_argument('--backup_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join_backup_51',
                       help='备份目录')
    
    args = parser.parse_args()
    
    statistics_join_sqls(args.backup_dir)

if __name__ == '__main__':
    main()

