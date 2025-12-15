#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计beijing和us数据集中单数据库和跨数据库SQL的数量
"""

import os
import json
from collections import defaultdict
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

def count_single_db_sqls(dataset_dir):
    """统计单数据库SQL数量"""
    single_dir = dataset_dir / "output" / "single"
    if not single_dir.exists():
        return 0
    
    count = 0
    for db_dir in single_dir.iterdir():
        if db_dir.is_dir():
            sql_files = list(db_dir.glob("generated_sql_*.json"))
            count += len(sql_files)
    
    return count

def count_cross_db_sqls(dataset_dir, cross_db_dirs):
    """统计跨数据库SQL数量，按数据库数量分类，区分有结果和无结果"""
    stats_with_results = defaultdict(int)  # {2: count, 3: count, 4: count}
    stats_without_results = defaultdict(int)  # {2: count, 3: count, 4: count}
    
    for cross_db_dir_name in cross_db_dirs:
        cross_db_dir = dataset_dir / "output" / cross_db_dir_name
        if not cross_db_dir.exists():
            continue
        
        # 查找所有JSON文件
        json_files = list(cross_db_dir.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查是否有结果
                results = data.get('results', [])
                has_results = results is not None and len(results) > 0
                
                # 方法1: 从metadata获取
                num_db = data.get('metadata', {}).get('num_databases')
                if num_db:
                    if has_results:
                        stats_with_results[num_db] += 1
                    else:
                        stats_without_results[num_db] += 1
                    continue
                
                # 方法2: 从databases字段获取
                databases = data.get('databases', [])
                if databases:
                    num_db = len(databases)
                    if has_results:
                        stats_with_results[num_db] += 1
                    else:
                        stats_without_results[num_db] += 1
                    continue
                
                # 方法3: 从table_database_mapping推断
                table_db_mapping = data.get('table_database_mapping', {})
                if table_db_mapping:
                    unique_dbs = set(table_db_mapping.values())
                    if unique_dbs:
                        num_db = len(unique_dbs)
                        if has_results:
                            stats_with_results[num_db] += 1
                        else:
                            stats_without_results[num_db] += 1
                        continue
                
                # 如果都找不到，尝试从文件名或内容推断
                # 默认认为是跨数据库的，但无法确定数量
                # print(f"警告: 无法确定 {json_file.name} 的数据库数量")
                
            except Exception as e:
                # print(f"错误: 读取 {json_file} 失败: {e}")
                pass
    
    return dict(stats_with_results), dict(stats_without_results)

def main():
    print("=" * 80)
    print("数据集SQL统计报告")
    print("=" * 80)
    print()
    
    # Beijing数据集
    print("【Beijing数据集】")
    print("-" * 80)
    beijing_dir = PROJECT_ROOT / "benchmark" / "data" / "beijing"
    
    # 单数据库SQL
    single_count = count_single_db_sqls(beijing_dir)
    print(f"单数据库SQL: {single_count:,} 个")
    
    # 跨数据库SQL
    cross_db_dirs = [
        "cross_db_single_join",
        "cross_db_single_join_backup_51",
        "cross_db_single",
        "cross_db_final"
    ]
    
    cross_db_stats_with, cross_db_stats_without = count_cross_db_sqls(beijing_dir, cross_db_dirs)
    
    total_cross_with = sum(cross_db_stats_with.values())
    total_cross_without = sum(cross_db_stats_without.values())
    total_cross = total_cross_with + total_cross_without
    
    print(f"跨数据库SQL总数: {total_cross:,} 个")
    print(f"  有结果: {total_cross_with:,} 个")
    print(f"  无结果: {total_cross_without:,} 个")
    
    if cross_db_stats_with or cross_db_stats_without:
        print("\n按数据库数量分布:")
        all_db_counts = set(list(cross_db_stats_with.keys()) + list(cross_db_stats_without.keys()))
        for num_db in sorted(all_db_counts):
            count_with = cross_db_stats_with.get(num_db, 0)
            count_without = cross_db_stats_without.get(num_db, 0)
            count_total = count_with + count_without
            if count_total > 0:
                print(f"  跨{num_db}个数据库: {count_total:,} 个 (有结果: {count_with:,}, 无结果: {count_without:,})")
    else:
        print("  (未找到跨数据库SQL)")
    
    print()
    
    # US数据集
    print("【US数据集】")
    print("-" * 80)
    us_dir = PROJECT_ROOT / "benchmark" / "data" / "us"
    
    # 单数据库SQL
    single_count_us = count_single_db_sqls(us_dir)
    print(f"单数据库SQL: {single_count_us:,} 个")
    
    # 跨数据库SQL
    cross_db_stats_us_with, cross_db_stats_us_without = count_cross_db_sqls(us_dir, cross_db_dirs)
    
    total_cross_us_with = sum(cross_db_stats_us_with.values())
    total_cross_us_without = sum(cross_db_stats_us_without.values())
    total_cross_us = total_cross_us_with + total_cross_us_without
    
    print(f"跨数据库SQL总数: {total_cross_us:,} 个")
    if total_cross_us > 0:
        print(f"  有结果: {total_cross_us_with:,} 个")
        print(f"  无结果: {total_cross_us_without:,} 个")
    
    if cross_db_stats_us_with or cross_db_stats_us_without:
        print("\n按数据库数量分布:")
        all_db_counts_us = set(list(cross_db_stats_us_with.keys()) + list(cross_db_stats_us_without.keys()))
        for num_db in sorted(all_db_counts_us):
            count_with = cross_db_stats_us_with.get(num_db, 0)
            count_without = cross_db_stats_us_without.get(num_db, 0)
            count_total = count_with + count_without
            if count_total > 0:
                print(f"  跨{num_db}个数据库: {count_total:,} 个 (有结果: {count_with:,}, 无结果: {count_without:,})")
    else:
        print("  (未找到跨数据库SQL)")
    
    print()
    
    # 总计
    print("=" * 80)
    print("【总计】")
    print("-" * 80)
    print(f"单数据库SQL总数: {single_count + single_count_us:,} 个")
    print(f"  其中 Beijing: {single_count:,} 个")
    print(f"  其中 US: {single_count_us:,} 个")
    print()
    print(f"跨数据库SQL总数: {total_cross + total_cross_us:,} 个")
    print(f"  其中 Beijing: {total_cross:,} 个 (有结果: {total_cross_with:,}, 无结果: {total_cross_without:,})")
    print(f"  其中 US: {total_cross_us:,} 个 (有结果: {total_cross_us_with:,}, 无结果: {total_cross_us_without:,})")
    print()
    
    # 合并统计（只统计有结果的）
    all_cross_stats_with = defaultdict(int)
    all_cross_stats_without = defaultdict(int)
    
    for num_db, count in cross_db_stats_with.items():
        all_cross_stats_with[num_db] += count
    for num_db, count in cross_db_stats_without.items():
        all_cross_stats_without[num_db] += count
    for num_db, count in cross_db_stats_us_with.items():
        all_cross_stats_with[num_db] += count
    for num_db, count in cross_db_stats_us_without.items():
        all_cross_stats_without[num_db] += count
    
    if all_cross_stats_with or all_cross_stats_without:
        print("跨数据库SQL按数量分布（合并，只统计有结果的）:")
        all_db_counts = set(list(all_cross_stats_with.keys()) + list(all_cross_stats_without.keys()))
        for num_db in sorted(all_db_counts):
            count_with = all_cross_stats_with.get(num_db, 0)
            count_without = all_cross_stats_without.get(num_db, 0)
            count_total = count_with + count_without
            if count_total > 0:
                print(f"  跨{num_db}个数据库: {count_with:,} 个 (总计: {count_total:,} 个)")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()

