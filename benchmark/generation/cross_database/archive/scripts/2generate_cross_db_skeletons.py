#!/usr/bin/env python3
"""
将单数据库SQL转换为跨数据库SQL骨架
"""

import os
import json
import re
import argparse
import random
from collections import defaultdict

def extract_tables_from_sql(sql):
    """从SQL中提取表名"""
    # 简单的表名提取（可以从FROM和JOIN子句中提取）
    tables = []
    
    # 提取FROM后的表
    from_match = re.search(r'FROM\s+["\']?(\w+)["\']?', sql, re.IGNORECASE)
    if from_match:
        tables.append(from_match.group(1))
    
    # 提取JOIN后的表
    join_matches = re.finditer(r'JOIN\s+["\']?(\w+)["\']?', sql, re.IGNORECASE)
    for match in join_matches:
        tables.append(match.group(1))
    
    return list(set(tables))  # 去重

def assign_tables_to_databases(tables, target_databases, strategy='round_robin'):
    """将表分配到不同数据库"""
    table_db_mapping = {}
    
    if strategy == 'round_robin':
        # 轮询分配，确保至少使用2个数据库
        for i, table in enumerate(tables):
            db_idx = i % min(len(target_databases), len(tables))
            table_db_mapping[table] = target_databases[db_idx]
    elif strategy == 'random':
        # 随机分配，确保至少使用2个数据库
        for i, table in enumerate(tables):
            if i < 2:
                # 前2个表分配到不同数据库
                db_idx = i % len(target_databases)
            else:
                # 后续表随机分配
                db_idx = random.randint(0, len(target_databases) - 1)
            table_db_mapping[table] = target_databases[db_idx]
    
    return table_db_mapping

def convert_to_cross_database_sql(original_sql, table_db_mapping):
    """将单数据库SQL转换为跨数据库SQL（添加数据库前缀）"""
    # 注意：这一步只是为了记录跨数据库SQL的格式，实际填充时由大模型完成
    converted_sql = original_sql
    
    # 按表名长度降序排序，避免短表名被长表名包含
    sorted_tables = sorted(table_db_mapping.keys(), key=len, reverse=True)
    
    for table in sorted_tables:
        db = table_db_mapping[table]
        # 替换表名：table -> database.table
        # 考虑表名可能带引号的情况
        patterns = [
            rf'\b{table}\b',  # 普通表名
            rf'"{table}"',    # 双引号
            rf"'{table}'",    # 单引号
        ]
        
        for pattern in patterns:
            replacement = f'{db}.{table}'
            converted_sql = re.sub(pattern, replacement, converted_sql, flags=re.IGNORECASE)
    
    return converted_sql

def convert_to_skeleton(sql):
    """将SQL转换为SQL骨架（使用原有的简单逻辑，不保留数据库前缀）"""
    # 使用原有的简单骨架生成逻辑
    sql_skeleton = re.sub(r"'[^']*'", '_', sql)
    sql_skeleton = re.sub(r'"[^"]*"', '_', sql_skeleton)
    sql_skeleton = re.sub(r'\b\d+\b', '_', sql_skeleton)
    sql_keywords = set(['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'JOIN', 'ON', 'AS', 'AND', 'OR', 'IN', 'NOT', 'NULL', 'IS', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER', 'UNION', 'ALL'])
    def replace_identifier(match):
        word = match.group(0)
        if word.upper() in sql_keywords or word == '*':
            return word
        else:
            return '_'
    sql_skeleton = re.sub(r'\b\w+\b', replace_identifier, sql_skeleton)
    sql_skeleton = re.sub(r'(_\s*)+', '_ ', sql_skeleton)
    sql_skeleton = ' '.join(sql_skeleton.strip().split())
    return sql_skeleton

def generate_cross_database_skeletons(candidates_file, target_databases, output_file):
    """生成跨数据库SQL骨架（简化版：骨架不包含数据库前缀，由后续填充步骤处理）"""
    # 加载候选SQL
    with open(candidates_file, 'r', encoding='utf-8') as f:
        candidates = json.load(f)
    
    cross_db_skeletons = []
    
    for candidate in candidates:
        original_sql = candidate['sql']
        tables = candidate['tables']
        
        # 分配表到数据库
        table_db_mapping = assign_tables_to_databases(tables, target_databases)
        
        # 使用原始SQL生成骨架（不添加数据库前缀）
        skeleton = convert_to_skeleton(original_sql)
        
        # 记录跨数据库信息（供后续填充时使用）
        cross_db_skeletons.append({
            'original_sql': original_sql,
            'original_database': candidate['database'],
            'original_file': candidate['file'],
            'sql_skeleton': skeleton,  # 普通骨架，不包含数据库前缀
            'databases': list(set(table_db_mapping.values())),  # 涉及的数据库列表
            'table_database_mapping': table_db_mapping,  # 表到数据库的映射
            'tables': tables,
            'table_count': len(tables),
            'is_cross_database': True,  # 标记为跨数据库查询
            'num_databases': len(set(table_db_mapping.values()))  # 涉及的数据库数量
        })
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cross_db_skeletons, f, ensure_ascii=False, indent=2)
    
    print(f"已生成 {len(cross_db_skeletons)} 个跨数据库SQL骨架")
    
    # 统计
    db_count_dist = defaultdict(int)
    for skeleton in cross_db_skeletons:
        db_count = skeleton['num_databases']
        db_count_dist[db_count] += 1
    
    print(f"\n跨数据库数量分布：")
    for db_count in sorted(db_count_dist.keys()):
        print(f"  跨{db_count}个数据库: {db_count_dist[db_count]}")

def main():
    parser = argparse.ArgumentParser(description='生成跨数据库SQL骨架')
    parser.add_argument('--candidates_file', type=str,
                       default='benchmark/generation/cross_database/candidates_2db.json',
                       help='候选SQL文件')
    parser.add_argument('--target_databases', type=str, nargs='+',
                       required=True,
                       help='目标数据库列表（例如：企业服务 社会保障）')
    parser.add_argument('--output_file', type=str,
                       default='benchmark/generation/cross_database/cross_db_skeletons_2db.json',
                       help='输出文件')
    parser.add_argument('--strategy', type=str, default='round_robin',
                       choices=['round_robin', 'random'],
                       help='表分配策略')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("生成跨数据库SQL骨架")
    print("=" * 70)
    print(f"\n候选文件: {args.candidates_file}")
    print(f"目标数据库: {args.target_databases}")
    print(f"输出文件: {args.output_file}")
    print(f"分配策略: {args.strategy}")
    print()
    
    generate_cross_database_skeletons(
        args.candidates_file,
        args.target_databases,
        args.output_file
    )
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()

