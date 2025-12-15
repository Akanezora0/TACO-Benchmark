#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查US数据集跨数据库SQL生成状态
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# 目标数量
TARGET_COUNTS = {
    2: 900,  # 跨2个数据库
    3: 264,  # 跨3个数据库
    4: 6     # 跨4个数据库
}

# 默认路径
DEFAULT_SQL_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output" / "cross_db_single_join"

def count_sqls_by_db_count(sql_dir):
    """统计各数据库数量的SQL（只统计有结果的）"""
    stats = defaultdict(int)  # {2: count, 3: count, 4: count}
    total = 0
    with_results = 0
    
    if not sql_dir.exists():
        return stats, total, with_results
    
    for sql_file in sql_dir.glob("cross_db_generated_sql_*.json"):
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 获取数据库数量
                num_db = data.get('metadata', {}).get('num_databases')
                if not num_db:
                    databases = data.get('databases', [])
                    if databases:
                        num_db = len(databases)
                    else:
                        table_db_mapping = data.get('table_database_mapping', {})
                        if table_db_mapping:
                            unique_dbs = set(table_db_mapping.values())
                            num_db = len(unique_dbs) if unique_dbs else 0
                
                if num_db in [2, 3, 4]:
                    total += 1
                    results = data.get('results', [])
                    if results is not None and len(results) > 0:
                        stats[num_db] += 1
                        with_results += 1
        except:
            pass
    
    return stats, total, with_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='检查US数据集跨数据库SQL生成状态')
    parser.add_argument('--sql-dir', type=str, default=None,
                       help=f'SQL文件目录（默认: {DEFAULT_SQL_DIR}）')
    
    args = parser.parse_args()
    
    sql_dir = Path(args.sql_dir) if args.sql_dir else DEFAULT_SQL_DIR
    
    print("=" * 80)
    print("US数据集跨数据库SQL生成状态")
    print("=" * 80)
    print()
    
    stats, total, with_results = count_sqls_by_db_count(sql_dir)
    
    print(f"总SQL文件数: {total}")
    print(f"有结果的SQL: {with_results} ({with_results/total*100:.1f}%)" if total > 0 else "有结果的SQL: 0")
    print()
    
    print("按数据库数量分布:")
    print("-" * 80)
    print(f"{'数据库数量':<15} {'有结果':<10} {'目标':<10} {'还需生成':<10} {'完成度':<10} {'状态':<10}")
    print("-" * 80)
    
    total_needed = 0
    total_with_results = 0
    total_target = 0
    
    for db_count in sorted([2, 3, 4]):
        with_result = stats.get(db_count, 0)
        target = TARGET_COUNTS[db_count]
        needed = max(0, target - with_result)
        
        total_with_results += with_result
        total_target += target
        total_needed += needed
        
        completion = (with_result / target * 100) if target > 0 else 0
        status = "✅ 完成" if with_result >= target else "⏳ 进行中"
        
        print(f"跨{db_count}个数据库  {with_result:<10} {target:<10} {needed:<10} {completion:>6.1f}%    {status:<10}")
    
    print("-" * 80)
    print(f"{'总计':<15} {total_with_results:<10} {total_target:<10} {total_needed:<10} {(total_with_results/total_target*100) if total_target > 0 else 0:>6.1f}%")
    print("=" * 80)
    
    if total_needed > 0:
        print(f"\n还需生成: {total_needed} 个有结果的SQL")
        print("建议运行: python3 run_all.py 或分步执行生成脚本")

if __name__ == '__main__':
    main()

