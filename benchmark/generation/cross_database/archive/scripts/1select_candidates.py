#!/usr/bin/env python3
"""
从单数据库SQL中选择适合扩展为跨数据库的候选SQL
"""

import os
import json
import argparse
import random

def load_single_database_sqls(sql_dir):
    """加载所有单数据库SQL"""
    all_sqls = []
    
    for db_name in sorted(os.listdir(sql_dir)):
        db_path = os.path.join(sql_dir, db_name)
        if not os.path.isdir(db_path):
            continue
        
        sql_files = [f for f in os.listdir(db_path) 
                     if f.startswith('generated_sql_') and f.endswith('.json') 
                     and not f.endswith('_error.json')]
        
        for sql_file in sql_files:
            sql_path = os.path.join(db_path, sql_file)
            try:
                with open(sql_path, 'r') as f:
                    sql_data = json.load(f)
                
                sql = sql_data.get('sql', '')
                tables = sql_data.get('tables', {})
                metadata = sql_data.get('metadata', {})
                
                # 检查是否包含JOIN且至少有2个表
                if 'JOIN' in sql.upper() and len(tables) >= 2:
                    all_sqls.append({
                        'database': db_name,
                        'file': sql_file,
                        'sql': sql,
                        'tables': list(tables.keys()),
                        'table_count': len(tables),
                        'has_join': metadata.get('has_join', False),
                        'has_subquery': metadata.get('has_subquery', False),
                        'sql_data': sql_data  # 保存完整数据
                    })
            except Exception as e:
                print(f"读取文件失败 {sql_path}: {e}")
                continue
    
    return all_sqls

def select_candidates(all_sqls, num_candidates=200, min_tables=2, max_tables=5):
    """选择候选SQL"""
    # 过滤：至少2个表，最多max_tables个表
    filtered = [s for s in all_sqls 
                if min_tables <= s['table_count'] <= max_tables]
    
    # 随机选择
    if len(filtered) >= num_candidates:
        selected = random.sample(filtered, num_candidates)
    else:
        selected = filtered
        # 如果不足，重复选择
        while len(selected) < num_candidates:
            selected.extend(filtered)
        selected = selected[:num_candidates]
    
    return selected

def save_candidates(candidates, output_file):
    """保存候选SQL"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    
    print(f"已保存 {len(candidates)} 条候选SQL到 {output_file}")

def main():
    parser = argparse.ArgumentParser(description='选择跨数据库SQL候选')
    parser.add_argument('--sql_dir', type=str, 
                       default='benchmark/data/beijing/output/single',
                       help='单数据库SQL目录')
    parser.add_argument('--output_file', type=str,
                       default='benchmark/generation/cross_database/candidates_2db.json',
                       help='输出文件')
    parser.add_argument('--num_candidates', type=int, default=200,
                       help='选择候选数量（默认200）')
    parser.add_argument('--min_tables', type=int, default=2,
                       help='最少表数量（默认2）')
    parser.add_argument('--max_tables', type=int, default=5,
                       help='最多表数量（默认5）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认42）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    print("=" * 70)
    print("选择跨数据库SQL候选")
    print("=" * 70)
    print(f"\nSQL目录: {args.sql_dir}")
    print(f"输出文件: {args.output_file}")
    print(f"候选数量: {args.num_candidates}")
    print(f"表数量范围: {args.min_tables}-{args.max_tables}")
    print()
    
    # 加载SQL
    print("加载单数据库SQL...")
    all_sqls = load_single_database_sqls(args.sql_dir)
    print(f"  找到 {len(all_sqls)} 条包含JOIN的SQL")
    
    # 选择候选
    print(f"\n选择候选SQL...")
    candidates = select_candidates(
        all_sqls, 
        args.num_candidates, 
        args.min_tables, 
        args.max_tables
    )
    print(f"  选择了 {len(candidates)} 条候选SQL")
    
    # 统计信息
    table_count_dist = {}
    for sql in candidates:
        count = sql['table_count']
        table_count_dist[count] = table_count_dist.get(count, 0) + 1
    
    print(f"\n候选SQL表数量分布：")
    for count in sorted(table_count_dist.keys()):
        print(f"  {count}个表: {table_count_dist[count]}")
    
    # 保存
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    save_candidates(candidates, args.output_file)
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()


