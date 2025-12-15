#!/usr/bin/env python3
"""
统计有结果的SQL数量

按2、3、4个数据库分类统计，并显示目标数量
"""

import os
import json
import argparse
from collections import defaultdict

# 目标数量（根据论文统计）
TARGET_COUNTS = {
    2: 359,  # 跨2个数据库
    3: 105,  # 跨3个数据库
    4: 2     # 跨4个数据库
}

def check_status(sql_dir):
    """统计有结果的SQL数量"""
    
    stats_by_db_count = defaultdict(lambda: {'total': 0, 'with_results': 0, 'without_results': 0})
    all_files = []
    
    print("统计SQL文件...")
    for f in sorted(os.listdir(sql_dir)):
        if f.startswith('cross_db_generated_sql_') and f.endswith('.json'):
            file_path = os.path.join(sql_dir, f)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    results = data.get('results', [])
                    num_databases = data.get('metadata', {}).get('num_databases', 
                                                                    len(data.get('databases', [])))
                    
                    stats_by_db_count[num_databases]['total'] += 1
                    
                    if results is not None and len(results) > 0:
                        stats_by_db_count[num_databases]['with_results'] += 1
                        all_files.append((num_databases, f))
                    else:
                        stats_by_db_count[num_databases]['without_results'] += 1
            except Exception as e:
                print(f"处理文件 {f} 时出错: {e}")
    
    # 显示统计结果
    print("\n" + "=" * 70)
    print("SQL生成状态统计")
    print("=" * 70)
    
    total_with_results = 0
    total_without_results = 0
    total_target = 0
    
    for db_count in sorted(stats_by_db_count.keys()):
        stats = stats_by_db_count[db_count]
        target = TARGET_COUNTS.get(db_count, 0)
        total_target += target
        
        with_results = stats['with_results']
        without_results = stats['without_results']
        total = stats['total']
        
        total_with_results += with_results
        total_without_results += without_results
        
        progress = (with_results / target * 100) if target > 0 else 0
        
        print(f"\n跨{db_count}个数据库:")
        print(f"  有结果: {with_results} / {target} ({progress:.1f}%)")
        print(f"  无结果: {without_results}")
        print(f"  总计: {total}")
        
        if with_results < target:
            print(f"  ⚠️  还需要: {target - with_results} 个")
        else:
            print(f"  ✅ 已完成目标")
    
    print(f"\n总计:")
    print(f"  有结果: {total_with_results} / {total_target} ({total_with_results/total_target*100:.1f}%)")
    print(f"  无结果: {total_without_results}")
    print(f"  总计: {total_with_results + total_without_results}")
    
    # 显示文件索引范围
    if all_files:
        print(f"\n文件索引范围:")
        for db_count in sorted(set(f[0] for f in all_files)):
            files = [f[1] for f in all_files if f[0] == db_count]
            indices = []
            for f in files:
                try:
                    idx = int(f.replace('cross_db_generated_sql_', '').replace('.json', ''))
                    indices.append(idx)
                except:
                    pass
            
            if indices:
                print(f"  {db_count}个数据库: {min(indices)} - {max(indices)} ({len(indices)} 个文件)")
    
    print("\n" + "=" * 70)
    
    return stats_by_db_count

def main():
    parser = argparse.ArgumentParser(description='统计有结果的SQL数量')
    parser.add_argument('--sql_dir', type=str,
                       default='benchmark/data/beijing/output/cross_db_single_join',
                       help='SQL文件目录')
    
    args = parser.parse_args()
    
    check_status(args.sql_dir)

if __name__ == '__main__':
    main()


