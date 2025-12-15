#!/usr/bin/env python3
"""
检查跨数据库SQL生成的目标完成情况
统计JOIN和UNION两种方式的总完成数量，并与目标对比
"""

import os
import json
from collections import defaultdict

# 目标数量（从README.md中获取）
TARGET_COUNTS = {
    2: 359,  # 跨2个数据库
    3: 105,  # 跨3个数据库
    4: 2     # 跨4个数据库
}

def count_sqls_with_results_in_directory(sql_dir, is_backup=False):
    """统计目录中有结果的SQL文件数量，按数据库数量分类"""
    stats = defaultdict(int)
    
    if not os.path.exists(sql_dir):
        return stats
    
    for filename in os.listdir(sql_dir):
        if not filename.startswith('cross_db_generated_sql_') or not filename.endswith('.json'):
            continue
        
        file_path = os.path.join(sql_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results = data.get('results', [])
            databases = data.get('databases', [])
            num_databases = len(databases)
            
            # 如果metadata中有num_databases，优先使用
            metadata = data.get('metadata', {})
            if 'num_databases' in metadata:
                num_databases = metadata['num_databases']
            
            if num_databases < 2 or num_databases > 4:
                continue
            
            # 判断是否有结果
            if is_backup:
                # 备份目录中的文件默认都是有结果的
                has_results = True
            else:
                has_results = results is not None and len(results) > 0
            
            if has_results:
                stats[num_databases] += 1
                
        except Exception as e:
            print(f"警告: 无法读取文件 {filename}: {e}")
            continue
    
    return stats

def main():
    # 获取脚本所在目录和项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    # 定义目录路径
    base_output_dir = os.path.join(project_root, 'benchmark', 'data', 'beijing', 'output')
    
    # JOIN相关目录
    join_dir = os.path.join(base_output_dir, 'cross_db_single_join')
    join_backup_dir = os.path.join(base_output_dir, 'cross_db_single_join_backup_51')
    
    # UNION相关目录
    union_dir = os.path.join(base_output_dir, 'cross_db_single_union_version')
    
    print("=" * 80)
    print("跨数据库SQL生成目标完成情况统计")
    print("=" * 80)
    print()
    
    # 统计JOIN方式（当前目录）
    print("📊 统计JOIN方式（当前目录）...")
    join_stats = count_sqls_with_results_in_directory(join_dir, is_backup=False)
    
    # 统计JOIN方式（备份目录）
    print("📊 统计JOIN方式（备份目录）...")
    join_backup_stats = count_sqls_with_results_in_directory(join_backup_dir, is_backup=True)
    
    # 合并JOIN统计
    join_total_stats = defaultdict(int)
    for db_count in [2, 3, 4]:
        join_total_stats[db_count] = join_stats[db_count] + join_backup_stats[db_count]
    
    # 统计UNION方式
    print("📊 统计UNION方式...")
    union_stats = count_sqls_with_results_in_directory(union_dir, is_backup=False)
    
    # 合并所有方式（JOIN + UNION）
    total_stats = defaultdict(int)
    for db_count in [2, 3, 4]:
        total_stats[db_count] = join_total_stats[db_count] + union_stats[db_count]
    
    # 输出结果
    print("\n" + "=" * 80)
    print("目标完成情况")
    print("=" * 80)
    print()
    
    print(f"{'数据库数量':<15} {'目标数量':<15} {'JOIN完成':<15} {'UNION完成':<15} {'总计完成':<15} {'完成率':<15} {'还需':<15}")
    print("-" * 80)
    
    total_target = 0
    total_completed = 0
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        join_completed = join_total_stats[db_count]
        union_completed = union_stats[db_count]
        total_completed_count = total_stats[db_count]
        completion_rate = (total_completed_count / target * 100) if target > 0 else 0
        remaining = max(0, target - total_completed_count)
        
        total_target += target
        total_completed += total_completed_count
        
        print(f"{db_count}个数据库{'':<6} {target:<15} {join_completed:<15} {union_completed:<15} {total_completed_count:<15} {completion_rate:.1f}%{'':<10} {remaining:<15}")
    
    print("-" * 80)
    total_completion_rate = (total_completed / total_target * 100) if total_target > 0 else 0
    total_remaining = max(0, total_target - total_completed)
    print(f"{'总计':<15} {total_target:<15} {sum(join_total_stats.values()):<15} {sum(union_stats.values()):<15} {total_completed:<15} {total_completion_rate:.1f}%{'':<10} {total_remaining:<15}")
    print()
    
    # 详细统计
    print("=" * 80)
    print("详细统计")
    print("=" * 80)
    print()
    
    for db_count in [2, 3, 4]:
        target = TARGET_COUNTS[db_count]
        join_completed = join_total_stats[db_count]
        union_completed = union_stats[db_count]
        total_completed_count = total_stats[db_count]
        completion_rate = (total_completed_count / target * 100) if target > 0 else 0
        remaining = max(0, target - total_completed_count)
        
        print(f"【{db_count}个数据库】")
        print(f"  目标数量: {target}")
        print(f"  JOIN完成: {join_completed} (当前目录: {join_stats[db_count]}, 备份目录: {join_backup_stats[db_count]})")
        print(f"  UNION完成: {union_completed}")
        print(f"  总计完成: {total_completed_count}")
        print(f"  完成率: {completion_rate:.1f}%")
        print(f"  还需: {remaining}")
        print()
    
    # 总体统计
    print("=" * 80)
    print("总体统计")
    print("=" * 80)
    print()
    print(f"总目标: {total_target} 个")
    print(f"JOIN完成: {sum(join_total_stats.values())} 个")
    print(f"UNION完成: {sum(union_stats.values())} 个")
    print(f"总计完成: {total_completed} 个")
    print(f"总完成率: {total_completion_rate:.1f}%")
    print(f"还需完成: {total_remaining} 个")
    print()
    
    # 输出目录路径
    print("=" * 80)
    print("目录路径")
    print("=" * 80)
    print(f"JOIN当前目录: {join_dir}")
    print(f"JOIN备份目录: {join_backup_dir}")
    print(f"UNION目录: {union_dir}")
    print()

if __name__ == '__main__':
    main()



