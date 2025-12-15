#!/usr/bin/env python3
"""
检查跨数据库SQL生成状态
"""

import os
import json
from collections import defaultdict

def check_status():
    skeleton_dir = "benchmark/generation/cross_database/skeletons"
    graph_dir = "benchmark/data/beijing/output/cross_db_graph"
    sql_dir = "benchmark/data/beijing/output/cross_db_single"
    
    # 1. 统计骨架文件
    skeleton_stats = defaultdict(int)
    total_skeletons = 0
    
    if os.path.exists(skeleton_dir):
        for f in os.listdir(skeleton_dir):
            if f.endswith('_skeletons.json'):
                with open(os.path.join(skeleton_dir, f), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    if data:
                        first = data[0]
                        num_db = first.get('num_databases', len(first.get('databases', [])))
                        skeleton_stats[f"{num_db}db"] += len(data)
                        total_skeletons += len(data)
    
    # 2. 统计图文件
    total_graphs = 0
    if os.path.exists(graph_dir):
        total_graphs = len([f for f in os.listdir(graph_dir) 
                           if f.startswith('cross_db_graph_') and f.endswith('.json')])
    
    # 3. 统计SQL文件
    sql_stats = {
        '2db': {'total': 0, 'with_results': 0, 'without_results': 0},
        '3db': {'total': 0, 'with_results': 0, 'without_results': 0},
        '4db': {'total': 0, 'with_results': 0, 'without_results': 0},
        'unknown': {'total': 0, 'with_results': 0, 'without_results': 0},
    }
    
    if os.path.exists(sql_dir):
        for f in os.listdir(sql_dir):
            if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
                try:
                    with open(os.path.join(sql_dir, f), 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        databases = data.get('databases', [])
                        num_db = len(databases) if databases else 0
                        
                        if num_db == 2:
                            key = '2db'
                        elif num_db == 3:
                            key = '3db'
                        elif num_db == 4:
                            key = '4db'
                        else:
                            key = 'unknown'
                        
                        sql_stats[key]['total'] += 1
                        if data.get('results') and len(data.get('results', [])) > 0:
                            sql_stats[key]['with_results'] += 1
                        else:
                            sql_stats[key]['without_results'] += 1
                except:
                    pass
    
    # 4. 目标数量
    target_2db = 359
    target_3db = 105
    target_4db = 2
    target_total = target_2db + target_3db + target_4db
    
    # 打印报告
    print("=" * 70)
    print("跨数据库SQL生成状态报告")
    print("=" * 70)
    
    print(f"\n1. 骨架文件统计:")
    for key in sorted(skeleton_stats.keys()):
        print(f"   {key}: {skeleton_stats[key]}")
    print(f"   总计: {total_skeletons}")
    
    print(f"\n2. 图文件生成:")
    print(f"   已生成: {total_graphs}")
    print(f"   完成率: {total_graphs/total_skeletons*100:.1f}%" if total_skeletons > 0 else "   完成率: 0%")
    if total_graphs < total_skeletons:
        print(f"   ⚠️  还有 {total_skeletons - total_graphs} 个图文件未生成")
    
    print(f"\n3. SQL文件生成:")
    total_sqls = sum(s['total'] for s in sql_stats.values())
    total_with_results = sum(s['with_results'] for s in sql_stats.values())
    
    for key in ['2db', '3db', '4db']:
        s = sql_stats[key]
        if s['total'] > 0:
            print(f"   {key}: {s['total']} (有结果: {s['with_results']}, {s['with_results']/s['total']*100:.1f}%)")
    
    if sql_stats['unknown']['total'] > 0:
        print(f"   unknown: {sql_stats['unknown']['total']}")
    
    print(f"   总计: {total_sqls} (有结果: {total_with_results}, {total_with_results/total_sqls*100:.1f}%)" if total_sqls > 0 else "   总计: 0")
    
    print(f"\n4. 目标对比:")
    print(f"   目标: 2db={target_2db}, 3db={target_3db}, 4db={target_4db}, 总计={target_total}")
    print(f"   当前: 2db={sql_stats['2db']['total']}, 3db={sql_stats['3db']['total']}, 4db={sql_stats['4db']['total']}, 总计={total_sqls}")
    
    if total_sqls > 0:
        print(f"   进度: {total_sqls/target_total*100:.1f}%")
        print(f"   有结果的有效SQL: {total_with_results} / {target_total} ({total_with_results/target_total*100:.1f}%)")
    
    print("\n" + "=" * 70)

if __name__ == '__main__':
    check_status()


