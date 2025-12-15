#!/usr/bin/env python3
"""
详细统计跨数据库SQL中union和join两种方式的数量
包括备份目录中的有结果SQL
按数据库数量（2、3、4）和方式（union、join）分类统计
"""

import os
import json
from collections import defaultdict

def count_sqls_in_directory(sql_dir, is_backup=False):
    """统计目录中的SQL文件，按数据库数量和是否有结果分类
    
    Args:
        sql_dir: 目录路径
        is_backup: 是否为备份目录（备份目录中的文件默认都是有结果的）
    """
    stats = {
        'total': 0,
        'with_results': 0,
        'without_results': 0,
        'by_db_count': defaultdict(lambda: {'total': 0, 'with_results': 0, 'without_results': 0})
    }
    
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
            
            stats['total'] += 1
            stats['by_db_count'][num_databases]['total'] += 1
            
            if has_results:
                stats['with_results'] += 1
                stats['by_db_count'][num_databases]['with_results'] += 1
            else:
                stats['without_results'] += 1
                stats['by_db_count'][num_databases]['without_results'] += 1
                
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
    print("跨数据库SQL详细统计：UNION vs JOIN（包含备份目录）")
    print("=" * 80)
    print()
    
    # 统计JOIN方式（当前目录）
    print("📊 统计JOIN方式（当前目录）...")
    join_stats = count_sqls_in_directory(join_dir, is_backup=False)
    
    # 统计JOIN方式（备份目录）
    print("📊 统计JOIN方式（备份目录）...")
    join_backup_stats = count_sqls_in_directory(join_backup_dir, is_backup=True)
    
    # 合并JOIN统计
    join_total_stats = {
        'total': join_stats['total'] + join_backup_stats['total'],
        'with_results': join_stats['with_results'] + join_backup_stats['with_results'],
        'without_results': join_stats['without_results'] + join_backup_stats['without_results'],
        'by_db_count': defaultdict(lambda: {'total': 0, 'with_results': 0, 'without_results': 0})
    }
    
    for db_count in [2, 3, 4]:
        join_total_stats['by_db_count'][db_count]['total'] = (
            join_stats['by_db_count'][db_count]['total'] + 
            join_backup_stats['by_db_count'][db_count]['total']
        )
        join_total_stats['by_db_count'][db_count]['with_results'] = (
            join_stats['by_db_count'][db_count]['with_results'] + 
            join_backup_stats['by_db_count'][db_count]['with_results']
        )
        join_total_stats['by_db_count'][db_count]['without_results'] = (
            join_stats['by_db_count'][db_count]['without_results'] + 
            join_backup_stats['by_db_count'][db_count]['without_results']
        )
    
    # 统计UNION方式
    print("📊 统计UNION方式...")
    union_stats = count_sqls_in_directory(union_dir, is_backup=False)
    
    # 输出统计结果
    print("\n" + "=" * 80)
    print("详细统计结果")
    print("=" * 80)
    print()
    
    # 按数据库数量输出
    for db_count in [2, 3, 4]:
        print(f"【{db_count}个数据库】")
        print("-" * 80)
        
        # JOIN方式统计
        join_current_total = join_stats['by_db_count'][db_count]['total']
        join_current_with_results = join_stats['by_db_count'][db_count]['with_results']
        join_current_without_results = join_stats['by_db_count'][db_count]['without_results']
        
        join_backup_total = join_backup_stats['by_db_count'][db_count]['total']
        join_backup_with_results = join_backup_stats['by_db_count'][db_count]['with_results']
        
        join_total = join_total_stats['by_db_count'][db_count]['total']
        join_with_results = join_total_stats['by_db_count'][db_count]['with_results']
        join_without_results = join_total_stats['by_db_count'][db_count]['without_results']
        
        print(f"  JOIN方式:")
        print(f"    当前目录: 总数={join_current_total}, 有结果={join_current_with_results}, 无结果={join_current_without_results}")
        print(f"    备份目录: 总数={join_backup_total}, 有结果={join_backup_with_results}")
        print(f"    合并统计: 总数={join_total}, 有结果={join_with_results} ({join_with_results/join_total*100:.1f}%)" if join_total > 0 else "    合并统计: 总数=0")
        print(f"              无结果={join_without_results}")
        
        # UNION方式统计
        union_total = union_stats['by_db_count'][db_count]['total']
        union_with_results = union_stats['by_db_count'][db_count]['with_results']
        union_without_results = union_stats['by_db_count'][db_count]['without_results']
        
        print(f"  UNION方式:")
        print(f"    总数: {union_total}")
        print(f"    有结果: {union_with_results} ({union_with_results/union_total*100:.1f}%)" if union_total > 0 else "    有结果: 0")
        print(f"    无结果: {union_without_results}")
        
        # 合并统计（JOIN + UNION）
        total_all = join_total + union_total
        with_results_all = join_with_results + union_with_results
        without_results_all = join_without_results + union_without_results
        
        print(f"  合并统计（JOIN + UNION）:")
        print(f"    总数: {total_all}")
        print(f"    有结果: {with_results_all} ({with_results_all/total_all*100:.1f}%)" if total_all > 0 else "    有结果: 0")
        print(f"    无结果: {without_results_all}")
        print()
    
    # 总体统计
    print("=" * 80)
    print("总体统计（所有数据库数量合并）")
    print("=" * 80)
    print()
    
    # JOIN总体统计
    join_current_total_all = join_stats['total']
    join_current_with_results_all = join_stats['with_results']
    join_current_without_results_all = join_stats['without_results']
    
    join_backup_total_all = join_backup_stats['total']
    join_backup_with_results_all = join_backup_stats['with_results']
    
    join_total_all = join_total_stats['total']
    join_with_results_all = join_total_stats['with_results']
    join_without_results_all = join_total_stats['without_results']
    
    print(f"JOIN方式:")
    print(f"  当前目录: 总数={join_current_total_all}, 有结果={join_current_with_results_all}, 无结果={join_current_without_results_all}")
    print(f"  备份目录: 总数={join_backup_total_all}, 有结果={join_backup_with_results_all}")
    print(f"  合并统计: 总数={join_total_all}, 有结果={join_with_results_all} ({join_with_results_all/join_total_all*100:.1f}%)" if join_total_all > 0 else "  合并统计: 总数=0")
    print(f"            无结果={join_without_results_all}")
    print()
    
    # UNION总体统计
    union_total_all = union_stats['total']
    union_with_results_all = union_stats['with_results']
    union_without_results_all = union_stats['without_results']
    
    print(f"UNION方式:")
    print(f"  总数: {union_total_all}")
    print(f"  有结果: {union_with_results_all} ({union_with_results_all/union_total_all*100:.1f}%)" if union_total_all > 0 else "  有结果: 0")
    print(f"  无结果: {union_without_results_all}")
    print()
    
    # 最终合并统计（JOIN + UNION）
    total_all = join_total_all + union_total_all
    with_results_all = join_with_results_all + union_with_results_all
    without_results_all = join_without_results_all + union_without_results_all
    
    print(f"最终合并统计（JOIN + UNION）:")
    print(f"  总数: {total_all}")
    print(f"  有结果: {with_results_all} ({with_results_all/total_all*100:.1f}%)" if total_all > 0 else "  有结果: 0")
    print(f"  无结果: {without_results_all}")
    print()
    
    # 输出表格格式（只显示有结果的）
    print("=" * 80)
    print("统计表格（有结果的SQL数量）")
    print("=" * 80)
    print()
    print(f"{'方式':<15} {'2个数据库':<20} {'3个数据库':<20} {'4个数据库':<20} {'总计':<20}")
    print("-" * 80)
    print(f"{'JOIN(当前)':<15} {join_stats['by_db_count'][2]['with_results']:<20} {join_stats['by_db_count'][3]['with_results']:<20} {join_stats['by_db_count'][4]['with_results']:<20} {join_current_with_results_all:<20}")
    print(f"{'JOIN(备份)':<15} {join_backup_stats['by_db_count'][2]['with_results']:<20} {join_backup_stats['by_db_count'][3]['with_results']:<20} {join_backup_stats['by_db_count'][4]['with_results']:<20} {join_backup_with_results_all:<20}")
    print(f"{'JOIN(合计)':<15} {join_total_stats['by_db_count'][2]['with_results']:<20} {join_total_stats['by_db_count'][3]['with_results']:<20} {join_total_stats['by_db_count'][4]['with_results']:<20} {join_with_results_all:<20}")
    print(f"{'UNION':<15} {union_stats['by_db_count'][2]['with_results']:<20} {union_stats['by_db_count'][3]['with_results']:<20} {union_stats['by_db_count'][4]['with_results']:<20} {union_with_results_all:<20}")
    print(f"{'总计':<15} {join_total_stats['by_db_count'][2]['with_results'] + union_stats['by_db_count'][2]['with_results']:<20} {join_total_stats['by_db_count'][3]['with_results'] + union_stats['by_db_count'][3]['with_results']:<20} {join_total_stats['by_db_count'][4]['with_results'] + union_stats['by_db_count'][4]['with_results']:<20} {with_results_all:<20}")
    print()
    
    # 输出详细文件路径信息
    print("=" * 80)
    print("目录路径")
    print("=" * 80)
    print(f"JOIN当前目录: {join_dir}")
    print(f"JOIN备份目录: {join_backup_dir}")
    print(f"UNION目录: {union_dir}")
    print()

if __name__ == '__main__':
    main()



