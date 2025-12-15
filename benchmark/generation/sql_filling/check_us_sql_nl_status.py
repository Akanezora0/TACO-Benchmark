#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查US数据集的SQL和NL查询生成情况
"""

import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "benchmark" / "data" / "us" / "output"
SINGLE_DIR = OUTPUT_DIR / "single"
NL_DIR = OUTPUT_DIR / "nl_query"

def count_sqls(db_name):
    """统计SQL数量（只统计有results的）"""
    sql_path = SINGLE_DIR / db_name
    if not sql_path.exists():
        return 0
    
    count = 0
    for sql_file in sql_path.glob("generated_sql_*.json"):
        try:
            with open(sql_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 只统计有results的SQL
                if 'results' in data and data['results'] is not None:
                    count += 1
        except:
            pass
    
    return count

def count_nl_queries(db_name):
    """统计NL查询数量"""
    nl_path = NL_DIR / db_name
    if not nl_path.exists():
        return 0
    
    count = 0
    for nl_file in nl_path.glob("*.json"):
        try:
            with open(nl_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 检查是否有natural_language_query字段
                if 'natural_language_query' in data and data['natural_language_query']:
                    count += 1
        except:
            pass
    
    return count

def main():
    print("US数据集SQL和NL查询生成情况统计")
    print("=" * 100)
    
    # 获取所有数据库
    databases = []
    if SINGLE_DIR.exists():
        databases = sorted([d.name for d in SINGLE_DIR.iterdir() if d.is_dir()])
    
    if not databases:
        print("未找到任何数据库")
        return
    
    print(f"\n{'数据库名称':<50} {'SQL数量':<10} {'NL查询数量':<12} {'状态':<20}")
    print("-" * 100)
    
    total_sql = 0
    total_nl = 0
    completed_sql = 0
    completed_nl = 0
    target_count = 220  # 目标数量
    
    for db_name in databases:
        sql_count = count_sqls(db_name)
        nl_count = count_nl_queries(db_name)
        
        total_sql += sql_count
        total_nl += nl_count
        
        # 判断状态
        if sql_count >= target_count and nl_count >= target_count:
            status = "✅ 完整 (SQL+NL)"
            completed_sql += 1
            completed_nl += 1
        elif sql_count >= target_count:
            status = f"⏳ 仅SQL完成 (缺{target_count - nl_count}个NL)"
            completed_sql += 1
        elif nl_count >= target_count:
            status = f"⚠️  仅NL完成 (缺{target_count - sql_count}个SQL)"
            completed_nl += 1
        else:
            status = f"❌ 未完成 (SQL缺{target_count - sql_count}, NL缺{target_count - nl_count})"
        
        display_name = db_name[:47] + "..." if len(db_name) > 50 else db_name
        print(f"{display_name:<50} {sql_count:<10} {nl_count:<12} {status:<20}")
    
    print("-" * 100)
    print(f"{'总计':<50} {total_sql:<10} {total_nl:<12} {completed_sql}/{len(databases)} SQL完成, {completed_nl}/{len(databases)} NL完成")
    print("=" * 100)
    
    # 统计摘要
    print("\n统计摘要:")
    print(f"  - 总数据库数: {len(databases)}")
    print(f"  - SQL总数: {total_sql} (目标: {len(databases) * target_count})")
    print(f"  - NL查询总数: {total_nl} (目标: {len(databases) * target_count})")
    print(f"  - SQL完成度: {completed_sql}/{len(databases)} ({completed_sql/len(databases)*100:.1f}%)")
    print(f"  - NL完成度: {completed_nl}/{len(databases)} ({completed_nl/len(databases)*100:.1f}%)")
    
    # 找出需要生成NL查询的数据库
    need_nl = []
    for db_name in databases:
        sql_count = count_sqls(db_name)
        nl_count = count_nl_queries(db_name)
        if sql_count >= target_count and nl_count < target_count:
            need_nl.append((db_name, sql_count, nl_count, target_count - nl_count))
    
    if need_nl:
        print(f"\n需要生成NL查询的数据库 ({len(need_nl)} 个):")
        for db_name, sql_count, nl_count, need in need_nl:
            print(f"  - {db_name}: SQL {sql_count}, NL {nl_count}, 还需 {need} 个NL查询")

if __name__ == '__main__':
    main()

